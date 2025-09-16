import os

APP_TITLE = os.getenv("APP_TITLE", "DUSTBOY")
SECRET_KEY = os.getenv("SECRET_KEY", "")

##
DIISRF_API_URL = os.getenv("DIISRF_API_URL", "http://localhost:8080/diisrf")
DIISRF_API_KEY = os.getenv("DIISRF_API_KEY", "")


#### DATABASE SETTINGS ####
MONGODB_DB = os.getenv("MONGODB_DB", "dustboydb")
MONGODB_HOST = os.getenv("MONGODB_HOST", "localhost")
MONGODB_PORT = int(os.getenv("MONGODDB_PORT", "27017"))
MONGODB_USERNAME = os.getenv("MONGODB_USERNAME", "")
MONGODB_PASSWORD = os.getenv("MONGODB_PASSWORD", "")
