__version__ = "0.1.0"

import optparse
import os
import json

from flask import Flask

from .. import models
from . import views
from . import acl
from . import redis_rq
from mongoengine import get_connection

app = Flask(__name__)


def create_app():
    app.config.from_object("podsaad.default_settings")

    for key, value in os.environ.items():
        app.config[key] = value
        value = value.strip()

        if not value:
            continue

        if value.isdigit():
            app.config[key] = int(value)

        if value.strip() and value.strip()[0] in ["[", "{"]:
            # app.config[key] = json.loads(value)
            app.config[key] = value

    app.config.from_envvar("PODSAAD_SETTING", silent=True)

    models.init_db(app)
    views.register_blueprint(app)
    acl.init_acl(app)
    redis_rq.init_rq(app)


    # --- DEBUG: Test MongoDB connection ---
    try:
        conn = get_connection()
        db_name = app.config.get("MONGODB_DB", "(unknown)")
        print(f"[DEBUG] MongoDB connection successful → Database: {db_name}")
        print(f"[DEBUG] Host: {app.config.get('MONGODB_HOST')}:{app.config.get('MONGODB_PORT')}")
    except Exception as e:
        print("[DEBUG] MongoDB connection failed!")
        print(e)

    jinja_env = app.jinja_env
    jinja_env.add_extension("jinja2.ext.do")

    return app


def get_program_options(default_host="127.0.0.1", default_port="8080"):
    """
    Takes a flask.Flask instance and runs it. Parses
    command-line flags to configure the app.
    """

    # Set up the command-line options
    parser = optparse.OptionParser()
    parser.add_option(
        "-H",
        "--host",
        help="Hostname of the Flask app " + "[default %s]" % default_host,
        default=default_host,
    )
    parser.add_option(
        "-P",
        "--port",
        help="Port for the Flask app " + "[default %s]" % default_port,
        default=default_port,
    )

    # Two options useful for debugging purposes, but
    # a bit dangerous so not exposed in the help message.
    parser.add_option(
        "-c", "--config", dest="config", help=optparse.SUPPRESS_HELP, default=None
    )
    parser.add_option(
        "-d", "--debug", action="store_true", dest="debug", help=optparse.SUPPRESS_HELP
    )
    parser.add_option(
        "-p",
        "--profile",
        action="store_true",
        dest="profile",
        help=optparse.SUPPRESS_HELP,
    )

    options, _ = parser.parse_args()

    # If the user selects the profiling option, then we need
    # to do a little extra setup
    if options.profile:
        from werkzeug.middleware.profiler import ProfilerMiddleware

        app.config["PROFILE"] = True
        app.wsgi_app = ProfilerMiddleware(app.wsgi_app, restrictions=[30])
        options.debug = True

    return options
