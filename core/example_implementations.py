import openai
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.multi_modal_llms import MultiModalLLM
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.multi_modal_llms.azure_openai import AzureOpenAIMultiModal
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from core.interfaces import AbstractLlmChat, AbstractLlmChatFactory, AbstractModelFactory
from core.config_handler import ConfigHandler, ConfigPrefix


class ExampleLlmChat(AbstractLlmChat):
    """Example implementation of an LLM chat with Azure OpenAI API"""
    def __init__(
        self,
        client: openai.AzureOpenAI,
        deployment: str,
        max_tokens: int = 4000,
    ) -> None:
        self._client: openai.AzureOpenAI = client
        self._deployment_llm: str = deployment
        self._max_tokens: int = max_tokens

    def get_response(self, system_msg: str, query: str) -> str:
        response = self._client.chat.completions.create(
            model=self._deployment_llm,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": query},
            ],
            max_tokens=4000,
        )
        content = response.choices[0].message.content
        return content


class ExampleChatModelFactory(AbstractLlmChatFactory):
    """Example implementation of an LLM chat factory with Azure OpenAI API with Azure Identity"""
    def __init__(self, config_handler: ConfigHandler) -> None:
        self.cnf = config_handler.get_config(ConfigPrefix.CHAT)

    def get_llm_chat(self) -> AbstractLlmChat:
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )
        client = openai.AzureOpenAI(
            azure_endpoint=self.cnf.endpoint,
            azure_deployment=self.cnf.llm_deployment,
            azure_ad_token_provider=token_provider,
            api_version=self.cnf.api_version,
        )
        return ExampleLlmChat(client, self.cnf.llm_deployment)
    

class ExampleModelFactory(AbstractModelFactory):
    """Example implementation of a model factory with Azure OpenAI API with Azure Identity"""
    def __init__(self, config_handler: ConfigHandler) -> None:
        self.config_h = config_handler
    
    def get_llm_model(self, config_type: ConfigPrefix) -> BaseLLM:
        cnf = self.config_h.get_config(config_type)
        return AzureOpenAI(
            model=cnf.llm_model,
            deployment_name=cnf.llm_deployment,
            use_azure_ad=True,
            base_url=f"{cnf.endpoint}/openai/deployments/{cnf.llm_deployment}",
            api_version=cnf.api_version,
            max_tokens=2000,
            temperature=cnf.temperature,
        )

    def get_multimodal_llm_model(self, config_type: ConfigPrefix) -> MultiModalLLM:
        cnf = self.config_h.get_config(config_type)
        return AzureOpenAIMultiModal(
            model=cnf.llm_model,
            deployment_name=cnf.llm_deployment,
            use_azure_ad=True,
            azure_endpoint=cnf.endpoint,
            api_version=cnf.api_version,
            max_new_tokens=1000,
            temperature=cnf.temperature,
        )

    def get_embed_model(self, config_type: ConfigPrefix) -> BaseEmbedding:
        cnf = self.config_h.get_config(config_type)
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )
        return AzureOpenAIEmbedding(
            model=cnf.embed_model,
            deployment_name=cnf.embed_deployment,
            use_azure_ad=True,
            azure_ad_token_provider=token_provider,
            base_url=f"{cnf.endpoint}/openai/deployments/{cnf.embed_deployment}",
            api_version=cnf.api_version,
            max_tokens=2000,
        )