import os
import flask
import logging

logger = logging.getLogger(__name__)

settings = None


def get_settings():
    global settings

    if not settings:
        filename = os.environ.get("PODSAAD_SETTINGS", None)


        logger.debug(f"setting environment file: {filename}")

        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")

        settings = flask.config.Config(file_path)
        settings.from_object("podsaad.default_settings")
        settings.from_envvar("PODSAAD_SETTINGS", silent=True)

    return settings