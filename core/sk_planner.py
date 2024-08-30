from pathlib import Path
from typing import Tuple

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureChatPromptExecutionSettings,
)
from semantic_kernel.utils.settings import azure_openai_settings_from_dot_env
from semantic_kernel.planners import (
    FunctionCallingStepwisePlanner,
    FunctionCallingStepwisePlannerOptions,
)
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from core.config_handler import ConfigHandler
from core.example_implementations import ExampleChatModelFactory, ExampleModelFactory
from misc.scenes_enum import Scene
from plugins.plugins_factory import PluginsFactory


def get_sk_planner(
    path_to_data: Path, plugins_dotenv_path: Path, scene_choice: Scene
) -> Tuple[FunctionCallingStepwisePlanner, Kernel]:
    """
    Creates an instance of the SK planner and kernel for the given scene.

    Args:
        path_to_data (Path): path to the main data directory
        plugins_dotenv_path (Path): path to the plugins' dotenv file
        scene_choice (Scene): chosen scene
    
    Returns:
        Tuple[FunctionCallingStepwisePlanner, Kernel]: planner and kernel instances
    """
    config_handler = ConfigHandler(plugins_dotenv_path)
    chat_model_factory = ExampleChatModelFactory(config_handler)
    model_factory = ExampleModelFactory(config_handler)

    plugins_factory = PluginsFactory(model_factory, chat_model_factory)
    nav_plugin = plugins_factory.get_nav_plugin(
        Path(f"{path_to_data}/nav_data/navmesh.txt")
    )
    text_plugin = plugins_factory.get_text_plugin(
        persist_dir=Path(f".TEXT_DIR/{scene_choice.value}"),
        text_dir=Path(f"{path_to_data}/text_data"),
    )
    sql_plugin = plugins_factory.get_sql_plugin(
        Path(f"{path_to_data}/sql_data/sql_db_data.json"),
        Path(f".SQL_DIR/{scene_choice.value}"),
    )
    image_plugin = plugins_factory.get_image_plugin(
        persist_dir=Path(f".IMAGE_DIR/{scene_choice.value}"),
        image_dir=Path(f"{path_to_data}/img_data"),
    )

    kernel = Kernel()
    deployment, _, endpoint = azure_openai_settings_from_dot_env()
    service_id = "default"

    kernel.add_service(
        AzureChatCompletion(
            service_id=service_id,
            deployment_name=deployment,
            endpoint=endpoint,
            ad_token_provider=get_bearer_token_provider(
                DefaultAzureCredential(),
                "https://cognitiveservices.azure.com/.default",
            ),
        ),
    )

    kernel.add_plugin(text_plugin, plugin_name="text")
    kernel.add_plugin(nav_plugin, plugin_name="navigation")
    kernel.add_plugin(sql_plugin, plugin_name="sql")
    kernel.add_plugin(image_plugin, plugin_name="image")

    options = FunctionCallingStepwisePlannerOptions(
        max_iterations=10,
        min_iteration_time_ms=2000,
        max_tokens=10000,
        execution_settings=AzureChatPromptExecutionSettings(temperature=0.0),
    )
    planner = FunctionCallingStepwisePlanner(
        service_id=service_id,
        options=options,
    )

    return planner, kernel
