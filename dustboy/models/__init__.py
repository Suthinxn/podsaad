from flask_mongoengine import MongoEngine
from flask import Flask

from .users import User

__all__ = ["User"]

db = MongoEngine()

def init_db(app: Flask):
    db.init_app(app)