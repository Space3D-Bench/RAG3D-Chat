from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional
from dotenv import dotenv_values


class ConfigPrefix(Enum):
    """
    Enum storing prefixes of keys used in dotenv file for corresponding models.
    """
    TEXT: str = "TEXT"
    IMAGES: str = "IMG"
    NAVIGATION: str = "NAV"
    SQL: str = "SQL"
    CHAT: str = "CHAT"


@dataclass
class Config:
    """
    Dataclass storing configuration for a single plugin/chat.
    """
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None
    llm_deployment: Optional[str] = None
    llm_model: Optional[str] = None
    embed_deployment: Optional[str] = None
    embed_model: Optional[str] = None
    api_version: Optional[str] = None
    temperature: float = 0.1


class ConfigHandler:
    def __init__(self, env_file: Path) -> None:
        """
        Constructor
        
        Args:
            env_file (Path): path to the dotenv file containing configuration values
        """
        self._config: Dict[str, str | None] = dotenv_values(env_file)
        self._retrieved_configs: Dict[str, Config] = {}

    def get_config(self, params_type: ConfigPrefix) -> Config:
        """
        Method returning configuration for a plugin/chat, depending on the prefix.

        Args:
            params_type (ConfigPrefix): prefix of the keys used in dotenv file for the plugin/chat

        Returns:
            Config: configuration for the plugin/chat
        """
        if params_type.value not in self._retrieved_configs:
            config = Config(
                api_key=self._config.get(f"{params_type.value}_API_KEY", None),
                endpoint=self._config.get(f"{params_type.value}_ENDPOINT", None),
                llm_deployment=self._config.get(
                    f"{params_type.value}_LLM_DEPLOYMENT_NAME", None
                ),
                llm_model=self._config.get(f"{params_type.value}_LLM_MODEL_NAME", None),
                embed_deployment=self._config.get(
                    f"{params_type.value}_EMBED_DEPLOYMENT_NAME", None
                ),
                embed_model=self._config.get(
                    f"{params_type.value}_EMBED_MODEL_NAME", None
                ),
                api_version=self._config.get(f"{params_type.value}_API_VERSION", None),
                temperature=self._config.get(f"{params_type.value}_TEMPERATURE", 0.1),
            )

            self._retrieved_configs[params_type.value] = config

        return self._retrieved_configs[params_type.value]
