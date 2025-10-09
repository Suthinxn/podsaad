from flask_mongoengine import MongoEngine
from flask import Flask

import mongoengine as me

from .pm25_interpolated_119t import PM25Interpolated119t
from .pm25_interpolated_120t import PM25Interpolated120t
from .pm25_interpolated_43t import PM25Interpolated43t
from .pm25_interpolated_44t import PM25Interpolated44t
from .pm25_interpolated_78t import PM25Interpolated78t
from .pm25_interpolated_o28 import PM25Interpolatedo28
from .pm25_interpolated_o73 import PM25Interpolatedo73

__all__ = [
    "PM25Interpolated119t",
    "PM25Interpolated120t",
    "PM25Interpolated43t",
    "PM25Interpolated44t",
    "PM25Interpolated78t",
    "PM25Interpolatedo28",
    "PM25Interpolatedo73",
]

db = MongoEngine()


def init_db(app: Flask):
    db.init_app(app)


def init_mongoengine(settings):
    dbname = settings.get("MONGODB_DB")
    host = settings.get("MONGODB_HOST", "localhost")
    port = int(settings.get("MONGODB_PORT", "27017"))
    username = settings.get("MONGODB_USERNAME", "")
    password = settings.get("MONGODB_PASSWORD", "")

    me.connect(db=dbname, host=host, port=port, username=username, password=password, authentication_source="admin")
