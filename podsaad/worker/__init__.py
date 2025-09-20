from .server import WorkerServer
from podsaad.web.config import get_settings

def create_server():

    settings = get_settings()
    server = WorkerServer(settings)

    return server