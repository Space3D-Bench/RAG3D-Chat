from abc import ABC, abstractmethod

from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.multi_modal_llms import MultiModalLLM


class AbstractLlmChat(ABC):
    """Interface for an LLM chat"""

    @abstractmethod
    def get_response(self, system_msg: str, query: str) -> str:
        """
        Method for getting a response from the LLM chat based on the system message and user query.
        
        Args:
            system_msg (str): system message
            query (str): user query
        
        Returns:
            str: response from the LLM chat 
        """
        raise NotImplementedError


class AbstractLlmChatFactory(ABC):
    """Interface for an LLM chat factory"""

    @abstractmethod
    def get_llm_chat(self) -> AbstractLlmChat:
        """
        Method for getting an LLM chat instance.
        
        Returns:
            AbstractLlmChat: LLM chat instance
        """
        raise NotImplementedError


class AbstractModelFactory(ABC):
    """Interface for a LLM/embedding model factory"""

    @abstractmethod
    def get_llm_model(self, prefix: str) -> BaseLLM:
        """
        Method for getting an LLM model instance.
        
        Args:
            prefix (str): prefix corresponding to the plugin
        
        Returns:
            BaseLLM: LLM model instance
        """
        raise NotImplementedError

    @abstractmethod
    def get_embed_model(self, prefix: str) -> BaseEmbedding:
        """
        Method for getting an embedding model instance.

        Args:
            prefix (str): prefix corresponding to the plugin
        
        Returns:
            BaseEmbedding: embedding model instance
        """
        raise NotImplementedError

    @abstractmethod
    def get_multimodal_llm_model(self, prefix: str) -> MultiModalLLM:
        """
        Method for getting a multimodal LLM model instance.

        Args:
            prefix (str): prefix corresponding to the plugin
        
        Returns:
            MultiModalLLM: multimodal LLM model instance
        """
        raise NotImplementedError
