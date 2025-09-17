from flask_mongoengine import MongoEngine
from flask import Flask


__all__ = []

db = MongoEngine()

def init_db(app: Flask):
    db.init_app(app)