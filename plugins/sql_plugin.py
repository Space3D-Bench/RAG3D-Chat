import json
from pathlib import Path
from typing import Annotated, Dict
import logging

from sqlalchemy import text
from llama_index.core import ServiceContext
from llama_index.core.schema import TextNode
from llama_index.core.llms import ChatMessage
from llama_index.core.retrievers import SQLRetriever
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT
from llama_index.legacy.llms.base import BaseLLM
from llama_index.core.llms import ChatResponse
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.legacy.embeddings.base import BaseEmbedding
from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    InputComponent,
    FnComponent,
)
from llama_index.legacy.indices.struct_store.sql_retriever import (
    DefaultSQLParser,
    BaseSQLParser,
)
from llama_index.core import (
    SQLDatabase,
    VectorStoreIndex,
    load_index_from_storage,
    StorageContext,
    PromptTemplate,
)

from semantic_kernel.functions.kernel_function_decorator import kernel_function

from core.rag_sql_loader import load_sql_database, get_dict_to_index
from plugins.plugin_prompts import (
    SQL_FUN_PROMPT,
    SQL_IN_PROMPT,
    SQL_OUT_PROMPT,
    SQL_TABLE_CONTEXTS,
    SQL_MENTIONED_ELEMENTS_PROMPT,
    SQL_FUN_DIST_PROMPT,
    SQL_IN_DIST_PROMPT,
    SQL_DIST_DISCARD_PROMPT
)

logger = logging.getLogger("SQL")


class SqlPlugin:
    def __init__(
        self,
        llm: BaseLLM,
        embed_model: BaseEmbedding,
        json_file_path: Path,
        persist_dir: Path = Path(".SQL_DIR"),
        top_k: int = 5,
    ) -> None:
        """
        Constructor.

        Args:
            llm (BaseLLM): LLM responsible for generating SQL queries from
                natural language descriptions and answering questions based
                on SQL retrieved data
            embed_model (BaseEmbedding): embedding model used to index the unique
                classes in the SQL tables
            json_file_path (Path): path to a JSON file containing data
                for the tables in the database
            persist_dir (Path): path to a directory in which the storage context
                is persisted (or is to be persisted if it does not exist yet)
            top_k (int): number of most similar classes to retrieve from the vector
                stores for each table

        Returns:
            None
        """
        self._embed_model: BaseEmbedding = embed_model
        self._llm: BaseLLM = llm
        self._sql_db: SQLDatabase = load_sql_database(json_file_path)
        self._to_index: Dict[str, str] = get_dict_to_index()
        self._vector_index_dict: Dict[str, VectorStoreIndex] = self._index_columns(
            persist_dir
        )
        self._top_k: int = top_k
        self._parser: BaseSQLParser = DefaultSQLParser()

        self._query_engine = NLSQLTableQueryEngine(
            sql_database=self._sql_db, llm=llm, embed_model=self._embed_model
        )
        self._qp = self._get_query_pipeline()

    @kernel_function(description=SQL_FUN_PROMPT, name="Sql")
    def get_quantitative_response(
        self, query: Annotated[str, SQL_IN_PROMPT]
    ) -> Annotated[str, SQL_OUT_PROMPT]:
        """
        Responds to the natual language query in accordance to the data retrieved
        from the SQL database

        Args:
            query (str): natural language query requesting the quantitative data
                from SQL DB

        Returns:
            str: natural language response to the query based on the data from SQL DB
        """
        logger.info(f"Query: {query}")
        response = str(self._qp.run(query=query))
        response = response.removeprefix("assistant: ")
        logger.info(f"Response: {response}")
        return response
    
    @kernel_function(description=SQL_FUN_DIST_PROMPT, name="SqlDist")
    def get_distance_related_response(
        self, query: Annotated[str, SQL_IN_DIST_PROMPT]
    ) -> Annotated[str, SQL_OUT_PROMPT]:
        """
        Responds to the distance-related natual language query in accordance to 
        the data retrieved from the SQL database (closest/furthest objects,
        objects within certain radius etc.)

        Args:
            query (str): natural language query containing the 3D position of an object
                to compare with others and the description of the task objective

        Returns:
            str: natural language response to the query based on the data from SQL DB
        """
        logger.info(f"Query: {query}")
        response = self._query_engine.query(query + SQL_DIST_DISCARD_PROMPT)
        logger.info(f"Response: {response}")
        return response

    def _index_columns(self, persist_dir_path: Path) -> Dict[str, VectorStoreIndex]:
        """
        Indexes columns in the SQL database in accordance to the specified columns in
        self._to_index[table_name], so that the unique classes in these columns are
        stored in the vector stores.

        Args:
            persist_dir_path (Path): path to a directory in which the storage context
                is persisted (or is to be persisted if it does not exist yet)

        Returns:
            Dict[str, VectorStoreIndex]: dictionary with table names as keys and
                values being VectorStoreIndex (with unique class names as elements)
        """
        vector_index_dict = {}
        for table_name in self._sql_db.get_usable_table_names():
            persist_subdir = persist_dir_path / table_name
            if not persist_subdir.is_dir():
                q = f'SELECT DISTINCT {self._to_index[table_name]} FROM "{table_name}"'
                with self._sql_db.engine.connect() as conn:
                    cursor = conn.execute(text(q))
                    result = cursor.fetchall()
                nodes = [TextNode(text=tuple(row)[0]) for row in result]
                index = VectorStoreIndex(nodes, embed_model=self._embed_model)
                index.storage_context.persist(str(persist_subdir))
            else:
                storage_context = StorageContext.from_defaults(
                    persist_dir=str(persist_subdir)
                )
                index = load_index_from_storage(
                    storage_context, embed_model=self._embed_model
                )
            vector_index_dict[table_name] = index

        return vector_index_dict

    def _get_table_context_str(self, available_classes_str: Dict[str, str]) -> str:
        """
        Based on the provided available classes strings, creates a context string for
        each table in the SQL database.

        Args:
            available_classes_str (Dict[str, str]): dictionary with table names as keys
                and strings with available classes as values
        
        Returns:
            str: context string for each table in the SQL database
        """
        context_strs = []
        for table_name in self._sql_db.get_usable_table_names():
            table_info = self._sql_db.get_single_table_info(table_name)
            table_info += " Do not try to access columns that were not mentioned.\n"
            table_info += SQL_TABLE_CONTEXTS[table_name]
            table_info += available_classes_str.get(table_name, "")
            context_strs.append(table_info)
        return "\n\n".join(context_strs)

    def _get_relevant_classes(self, elements: ChatMessage) -> Dict[str, str]:
        """
        Retrieves classes from the SQL database which are most similar to the mentioned
        elements, extracted by a chat from the natural language query.

        Args:
            elements (ChatMessage): message containing the mentioned elements in a form
                of json
        
        Returns:
            Dict[str, str]: dictionary with table names as keys and strings with the
                most similar available classes as values
        """
        try:
            elements_dict: Dict[str, str] = json.loads(elements.message.content)
            columns_context = {}
            for table_name in elements_dict:
                vector_retriever = self._vector_index_dict[table_name].as_retriever(
                    similarity_top_k=self._top_k
                )
                available_classes = set()
                for el in elements_dict[table_name]:
                    relevant_nodes = vector_retriever.retrieve(el)
                    for node in relevant_nodes:
                        available_classes.add(node.get_content())
                logger.info(
                    f"Most similar classes to {elements_dict[table_name]}: {available_classes}"
                )

                columns_context[table_name] = ""
                if len(available_classes) > 0:
                    columns_context[table_name] += (
                        "\nWhen using the WHERE statement, under no circumstances you are" 
                        f" allowed to use other values for the '{self._to_index[table_name]}'"
                        f" column in table '{table_name}' than the following: "
                    )
                    columns_context[table_name] += (
                        ", ".join(f"'{class_name}'" for class_name in available_classes)
                        + "."
                    )
            return columns_context
        except:
            return {}        

    def _parse_sql_response(self, response: ChatResponse) -> str:
        """
        Parses the response from the SQL database.

        Args:
            response (ChatResponse): response from the SQL database
                (including the SQL query and the results)

        Returns:
            str: SQL query response
        """
        logger.info(f"SQL query: {response.message.content}")
        return self._parser.parse_response_to_sql(response.message.content, "")

    def _get_query_pipeline(self) -> QP:
        """
        Creates a query pipeline for answering natural language questions,
        based on the data of the SQL database.

        Returns:
            QP: LlamaIndex's query pipeline for the SQL plugin
        """

        sql_retriever = SQLRetriever(self._sql_db)

        context_parser_component = FnComponent(fn=self._get_table_context_str)
        sql_parser_component = FnComponent(fn=self._parse_sql_response)
        elements_retriever_component = FnComponent(fn=self._get_relevant_classes)

        text2sql_prompt = DEFAULT_TEXT_TO_SQL_PROMPT.partial_format(
            dialect=self._sql_db.engine.dialect.name
        )
        response_synthesis_prompt = PromptTemplate(
            template=(
                "Given an input question, synthesize a response from the query results.\n"
                "Query: {query_str}\n"
                "SQL: {sql_query}\n"
                "SQL Response: {context_str}\n"
                "Response: "
            )
        )

        mentioned_elements_prompt = PromptTemplate(
            template=(SQL_MENTIONED_ELEMENTS_PROMPT + "Query: {query_str}\n")
        )

        qp = QP(
            modules={
                "input": InputComponent(),
                "query2elements_prompt": mentioned_elements_prompt,
                "query2elements_llm": self._llm,
                "elements_retriever_parser": elements_retriever_component,
                "context_parser": context_parser_component,
                "text2sql_prompt": text2sql_prompt,
                "text2sql_llm": self._llm,
                "sql_output_parser": sql_parser_component,
                "sql_retriever": sql_retriever,
                "response_synthesis_prompt": response_synthesis_prompt,
                "response_synthesis_llm": self._llm,
            },
        )

        qp.add_link("input", "query2elements_prompt", dest_key="query_str")
        qp.add_link("query2elements_prompt", "query2elements_llm")
        qp.add_link(
            "query2elements_llm", "elements_retriever_parser", dest_key="elements"
        )
        qp.add_link(
            "elements_retriever_parser",
            "context_parser",
            dest_key="available_classes_str",
        )
        qp.add_link("input", "text2sql_prompt", dest_key="query_str")
        qp.add_link("context_parser", "text2sql_prompt", dest_key="schema")
        qp.add_chain(
            ["text2sql_prompt", "text2sql_llm", "sql_output_parser", "sql_retriever"]
        )
        qp.add_link(
            "sql_output_parser", "response_synthesis_prompt", dest_key="sql_query"
        )
        qp.add_link(
            "sql_retriever", "response_synthesis_prompt", dest_key="context_str"
        )
        qp.add_link("input", "response_synthesis_prompt", dest_key="query_str")
        qp.add_link("response_synthesis_prompt", "response_synthesis_llm")

        return qp
