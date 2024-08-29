import logging
from pathlib import Path
from typing import Annotated, Optional, Set

from llama_index.legacy.core.base_retriever import BaseRetriever
from llama_index.core.indices.base import BaseIndex
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.embeddings.multi_modal_base import MultiModalEmbedding
from llama_index.core.multi_modal_llms import MultiModalLLM

from semantic_kernel.functions.kernel_function_decorator import kernel_function

from core.rag_document_loaders import load_image_documents, local_image_to_document
from core.interfaces import AbstractLlmChat
from plugins.plugin_prompts import (
    IMAGE_FUN_PROMPT,
    IMAGE_IN_PROMPT,
    IMAGE_OUT_PROMPT,
    IMAGE_METADATA_PROMPT,
)

logger = logging.getLogger("IMAGE")


class ImagePlugin:
    def __init__(
        self,
        vision_llm: MultiModalLLM,
        llm_chat: AbstractLlmChat,
        embed_model: MultiModalEmbedding,
        image_dir: Optional[Path] = None,
        persist_dir: Path = Path(".IMAGE_DIR"),
    ) -> None:
        """
        Constructor

        Args:
            vision_llm (BaseLLM): LLM model answering the queries based on the attached image
            llm_chat (AbstractLlmChat): LLM chat used for filter definition
            embed_model (MultiModalEmbedding): multimodal embedding model
            image_dir (optional, Path): path to a directory in which the image data is stored
            persist_dir (Path): path to a directory in which the storage context
                is persisted (or is to be persisted if it does not exist yet)

        Returns:
            None
        """
        self._embed_model: MultiModalEmbedding = embed_model
        self._llm: MultiModalLLM = vision_llm
        self._rooms: Set[str] = set()
        self._index: BaseIndex = self._get_index(persist_dir, image_dir)
        self._llm_chat: AbstractLlmChat = llm_chat

    @kernel_function(description=IMAGE_FUN_PROMPT, name="Image")
    def get_visual_response(
        self, query: Annotated[str, IMAGE_IN_PROMPT]
    ) -> Annotated[str, IMAGE_OUT_PROMPT]:
        """
        Gets the response from the LLM for the defined query based on the retrieved image

        Args:
            query (str): a query regarding the stored data

        Returns:
            str: answer from the LLM
        """
        logger.info(f"Query: {query}")
        retriever = self._get_retiever(query)  # includes a filter
        img_nodes = retriever.retrieve(query)
        retrieved_img_path = img_nodes[0].metadata["file_path"]
        logger.info(f"Retrieved image: {retrieved_img_path}")

        img_doc = local_image_to_document(retrieved_img_path)
        complete_response = self._llm.complete(
            prompt=query
            + " Do not use 'left' and 'right' when describing the positions, since it depeneds on the point of view.",
            image_documents=[img_doc],
        )
        logger.info(f"Response: {complete_response}")
        return complete_response.text

    def _get_index(
        self, persist_dir: Path, image_dir: Optional[Path] = None
    ) -> BaseIndex:
        """
        Creates an index based on the data in the provided image directory or in
            the previously saved storage context.

        Args:
            persist_dir (Path): path to a directory in which the storage context
                is persisted (or is to be persisted if it does not exist yet)
            image_dir (optional, Path): path to a directory containing data for RAG (each
                folder having images for separate rooms)

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

            image_index = load_index_from_storage(
                storage_context, embed_model=self._embed_model
            )

        elif image_dir and image_dir.is_dir():
            img_documents = load_image_documents(image_dir)

            image_index = MultiModalVectorStoreIndex.from_documents(
                img_documents,
                embed_model=self._embed_model,
                is_text_vector_store_empty=True,
                show_progress=True,
            )

            for doc in img_documents:
                self._rooms.add(doc.metadata["room_name"])

            image_index.storage_context.persist(persist_dir=persist_dir)
        else:
            raise FileNotFoundError(
                "Neither the persist directory nor the text data directory exists."
            )

        return image_index

    def _get_retiever(self, query: str) -> BaseRetriever:
        """
        Creates a retriever, with a metadata filter if the query concerns
        a specific room.

        Args:
            query (str): a query regarding the stored data

        Returns:
            BaseRetriever: retriever for the image RAG
        """
        system_msg = IMAGE_METADATA_PROMPT.format(self._rooms)
        response = self._llm_chat.get_response(system_msg, query)

        if response == "None" or response not in self._rooms:
            logger.info("No room specified in the query.")
            return self._index.as_retriever(embed_model=self._embed_model)
        else:
            logger.info(f"Room specified in the query: {response}")
            filters = MetadataFilters(
                filters=[ExactMatchFilter(key="room_name", value=response)]
            )
            return self._index.as_retriever(
                embed_model=self._embed_model, filters=filters
            )
