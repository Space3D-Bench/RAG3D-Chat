import logging.config
from pathlib import Path
import logging

from core.config_handler import ConfigHandler
from core.example_implementations import ExampleChatModelFactory, ExampleModelFactory
from plugins.plugins_factory import PluginsFactory
from misc.scenes_enum import Scene

logging.config.fileConfig("conf/logging_conf.ini")
logger_main = logging.getLogger("main")
SCENE = Scene.APARTMENT_2  # change this to test different scenes


def test_sql_plugin(plugins_factory: PluginsFactory):
    sql_plugin = plugins_factory.get_sql_plugin(
        Path(f"data/{SCENE.value}/sql_data/sql_db_data.json"),
        Path(f".SQL_DIR/{SCENE.value}"),
    )
    logger_main.info("SQL PLUGIN")
    query = "Are there any plants in the apartment?"
    logger_main.info(query)
    logger_main.info(sql_plugin.get_quantitative_response(query))
    logger_main.info("=====================\n")


def test_img_plugin(plugins_factory: PluginsFactory):
    image_plugin = plugins_factory.get_image_plugin(
        image_dir=Path(f"data/{SCENE.value}/img_data"),
        persist_dir=Path(f".IMAGE_DIR/{SCENE.value}"),
    )
    logger_main.info("IMAGE PLUGIN")
    query = "Is there a place to sit in the room in which you can prepare dinner?"
    logger_main.info(query)
    logger_main.info(image_plugin.get_visual_response(query))
    logger_main.info("=====================\n")


def test_text_plugin(plugins_factory: PluginsFactory):
    text_plugin = plugins_factory.get_text_plugin(
        persist_dir=Path(f".TEXT_DIR/{SCENE.value}"),
        text_dir=Path(f"data/{SCENE.value}/txt_data"),
    )
    logger_main.info("TEXT PLUGIN")
    query = "Describe the living room."
    logger_main.info(query)
    logger_main.info(text_plugin.get_descriptive_response(query))
    logger_main.info("=====================\n")


def test_nav_plugin(plugins_factory: PluginsFactory):
    nav_plugin = plugins_factory.get_nav_plugin(
        Path(f"data/{SCENE.value}/nav_data/navmesh.txt")
    )

    logger_main.info("NAV PLUGIN")
    query = "Sofa is at position 4.720187, 2.382002, -1.22266, the bed is at 4.216630, 0.766252, -0.86965."
    logger_main.info(query)
    logger_main.info(nav_plugin.get_straight_line_distance_from_query(query))
    logger_main.info(nav_plugin.get_actual_distance_from_query(query))
    logger_main.info("=====================\n")
    query = "The mirror is at position (0.77, 0.01, -0.62), and the desk is at (-0.787, 0.0587, -1.22661)."
    logger_main.info(query)
    logger_main.info(nav_plugin.get_actual_distance_from_query(query))
    logger_main.info("=====================\n")


if __name__ == "__main__":
    ### adjust this part so that it corresponds to your implementations
    config_handler = ConfigHandler(Path("plugins_dotenv.env"))
    chat_model_factory = ExampleChatModelFactory(config_handler)
    model_factory = ExampleModelFactory(config_handler)
    ###

    plugins_factory = PluginsFactory(model_factory, chat_model_factory)
    test_text_plugin(plugins_factory)
    test_img_plugin(plugins_factory)
    test_sql_plugin(plugins_factory)
    test_nav_plugin(plugins_factory)
