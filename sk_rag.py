import asyncio
import json
import logging
import logging.config
from pathlib import Path

from core.sk_planner import get_sk_planner
from misc.scenes_enum import Scene


logging.config.fileConfig("conf/logging_conf.ini")
logger_plugins = logging.getLogger("plugins")
logger_main = logging.getLogger("main")


async def main():
    for scene_choice in Scene:
        try:
            path_to_data = Path(f"data/{scene_choice.value}")
            questions_path = path_to_data / "questions.json"
            plugins_dotenv = Path(".env_plugins")
            answers_path = Path(f"results/{scene_choice.value}/answers.json")

            separator_scene = f"===================== Testing {scene_choice.value} ====================="
            logger_plugins.info(separator_scene)
            logger_main.info(separator_scene)
            
            with questions_path.open("r") as file:
                questions = json.load(file)
            planner, kernel = get_sk_planner(path_to_data, plugins_dotenv, scene_choice)

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
                    result = await planner.invoke(kernel, q)
                    logger_main.info(result.final_answer)

                    logger_plugins.info(result.final_answer)
                    logger_plugins.info("---")
                    logger_plugins.info(result.chat_history[0])

                    answers[nr] = result.final_answer
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
