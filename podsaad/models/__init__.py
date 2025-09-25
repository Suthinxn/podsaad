from flask_mongoengine import MongoEngine
from flask import Flask

import mongoengine as me


__all__ = []

db = MongoEngine()


def init_db(app: Flask):
    db.init_app(app)


def init_mongoengine(settings):
    dbname = settings.get("MONGODB_DB")
    host = settings.get("MONGODB_HOST", "localhost")
    port = int(settings.get("MONGODB_PORT", "27017"))
    username = settings.get("MONGODB_USERNAME", "")
    password = settings.get("MONGODB_PASSWORD", "")

    me.connect(db=dbname, host=host, port=port, username=username, password=password)
