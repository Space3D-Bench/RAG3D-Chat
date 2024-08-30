import logging
from pathlib import Path
from typing import Annotated, Optional, List, Set

from llama_index.legacy.embeddings.base import BaseEmbedding
from llama_index.legacy.llms.base import BaseLLM
from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter
from llama_index.core.indices.base import BaseIndex
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)

from core.interfaces import AbstractLlmChat
from core.rag_document_loaders import load_text_documents
from plugins.plugin_prompts import (
    TEXT_FUN_PROMPT,
    TEXT_IN_PROMPT,
    TEXT_OUT_PROMPT,
    TEXT_METADATA_PROMPT,
)

logger = logging.getLogger("TEXT")


class TextPlugin:
    def __init__(
        self,
        llm: BaseLLM,
        embed_model: BaseEmbedding,
        llm_chat: AbstractLlmChat,
        txt_dir: Optional[Path] = None,
        persist_dir: Path = Path(".TEXT_DIR"),
        top_k: int = 3,
    ) -> None:
        """
        Constructor

        Args:
            llm (BaseLLM): LLM model answering the queries based on the stored data
            embed_model (BaseEmbedding): embedding model
            llm_chat (AbstractLlmChat): LLM chat used for filter definition
            txt_dir (optional, Path): path to a directory containing text file with data
                for RAG (each file being a separate document)
            persist_dir (Path): path to a directory in which the storage context
                is persisted (or is to be persisted if it does not exist yet)
            top_k (int): number of documents most similar to the query taken into account
                when answering the it

        Returns:
            None
        """
        self._top_k: int = top_k
        self._llm: BaseLLM = llm
        self._embed_model: BaseEmbedding = embed_model
        self._llm_chat: AbstractLlmChat = llm_chat
        self._rooms: Set[str] = set()
        self._index: BaseIndex = self._get_index(persist_dir, txt_dir)

    @kernel_function(description=TEXT_FUN_PROMPT, name="Text")
    def get_descriptive_response(
        self, query: Annotated[str, TEXT_IN_PROMPT]
    ) -> Annotated[str, TEXT_OUT_PROMPT]:
        """
        Gets the response from the LLM for the defined query based on the stored data

        Args:
            query (str): a query regarding the stored data

        Returns:
            str: answer from the LLM
        """
        logger.info(f"Query: {query}")
        query_engine = self._get_query_engine(query)
        result = query_engine.query(query)
        logger.info(f"Answer: {result}")
        return result.response

    def _get_index(
        self, persist_dir: Path, text_dir: Optional[Path] = None
    ) -> BaseIndex:
        """
        Creates an index based on the data in the provided text files or in the previously
        saved storage context.

        Args:
            persist_dir (Path): path to a directory in which the storage context
                is persisted (or is to be persisted if it does not exist yet)
            text_dir (optional, Path): path to a directory containing text file with data
                for RAG (each file being a separate document)

        Returns:
            BaseIndex: index for the RAG application based on the textual data

        Raises:
            FileNotFoundError: if neither the persist directory nor the input text file
                exists/is valid
        """

        if persist_dir.is_dir():
            storage_context: StorageContext = StorageContext.from_defaults(
                persist_dir=persist_dir
            )
            for doc in storage_context.docstore.docs.values():
                self._rooms.add(doc.metadata["room_name"])

            index: BaseIndex = load_index_from_storage(
                storage_context, embed_model=self._embed_model
            )

        elif text_dir and text_dir.is_dir():
            docs: List[Document] = load_text_documents(text_dir)
            index: BaseIndex = VectorStoreIndex.from_documents(
                docs, embed_model=self._embed_model
            )

            for doc in docs:
                self._rooms.add(doc.metadata["room_name"])
            index.storage_context.persist(persist_dir=persist_dir)
        else:
            raise FileNotFoundError(
                "Neither the persist directory nor the text data directory exists."
            )
        return index

    def _get_query_engine(self, query: str):
        """
        Creates a query engine, including a metadata filter based on the query

        Args:
            query (str): a query regarding the stored data

        Returns:
            BaseQueryEngine: query engine with the metadata filter if the query is about
            specific rooms
        """
        system_msg = TEXT_METADATA_PROMPT.format(self._rooms)
        response = self._llm_chat.get_response(system_msg, query)

        if response == "None" or response not in self._rooms:
            logger.info("No metadata filter")
            return self._index.as_query_engine(
                llm=self._llm,
                embed_model=self._embed_model,
                similarity_top_k=self._top_k,
            )
        else:
            logger.info(f"Rooms specified in the query: {response}")
            filters = MetadataFilters(
                filters=[ExactMatchFilter(key="room_name", value=response)]
            )
            return self._index.as_query_engine(
                llm=self._llm,
                embed_model=self._embed_model,
                similarity_top_k=self._top_k,
                filters=filters,
            )
