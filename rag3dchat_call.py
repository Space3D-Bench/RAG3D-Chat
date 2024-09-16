import asyncio
import json
import logging
import logging.config
from pathlib import Path

from semantic_kernel.utils.settings import azure_openai_settings_from_dot_env
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

from core.rag3dchat import RAG3DChat
from core.config_handler import ConfigHandler
from core.example_implementations import ExampleChatModelFactory, ExampleModelFactory
from misc.scenes_enum import Scene
from plugins.plugins_factory import PluginsFactory


logging.config.fileConfig("conf/logging_conf.ini")
logger_plugins = logging.getLogger("plugins")
logger_main = logging.getLogger("main")


async def main():
    ### adjust this part so that it corresponds to your implementations
    plugins_dotenv = Path(".env_plugins")
    config_handler = ConfigHandler(plugins_dotenv)
    chat_model_factory = ExampleChatModelFactory(config_handler)
    model_factory = ExampleModelFactory(config_handler)
    
    deployment, _, endpoint = azure_openai_settings_from_dot_env()
    kernel_service = AzureChatCompletion(
        service_id="default",
        deployment_name=deployment,
        endpoint=endpoint,
        ad_token_provider=get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default",
        ),
    )
    ###
    plugins_factory = PluginsFactory(model_factory, chat_model_factory)
    
    for scene_choice in Scene:
        try:
            path_to_data = Path(f"data")
            questions_path = path_to_data / "questions.json"
            answers_path = path_to_data / "answers.json"

            if questions_path.exists() is False:
                logger_main.info(f"No questions found for {scene_choice.value}")
                continue
            
            separator_scene = f"===================== Testing {scene_choice.value} ====================="
            logger_plugins.info(separator_scene)
            logger_main.info(separator_scene)
            
            with questions_path.open("r") as file:
                questions = json.load(file)

            rag3dchat = RAG3DChat(plugins_factory, path_to_data)
            rag3dchat.set_scene(scene_choice)
            rag3dchat.set_sk(kernel_service)

            answers = {}
            if answers_path.exists():
                with answers_path.open("r") as file:
                    answers = json.load(file)

            for nr, q in questions.items():
                if nr in answers:
                    continue
                separator = "======================="
                question = f"{nr}. {q}"

                logger_plugins.info(separator)
                logger_plugins.info(question)
                logger_main.info(separator)
                logger_main.info(question)

                try:
                    final_answer, generated_plan = await rag3dchat.get_answer(q)
                    logger_main.info(final_answer)

                    logger_plugins.info(final_answer)
                    logger_plugins.info("---")
                    logger_plugins.info(generated_plan)

                    answers[nr] = final_answer
                    with answers_path.open("w") as file:
                        json.dump(answers, file, indent=4)

                except Exception as e:
                    error_str = f"Error with question {q}: {e}"
                    logger_plugins.error(error_str)
                    logger_main.error(error_str)

        except Exception as e:
            print(f"Tests for {scene_choice.value} interrupted: {e}")


if __name__ == "__main__":
    asyncio.run(main())
