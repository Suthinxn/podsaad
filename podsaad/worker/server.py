import redis
from rq import Queue, SimpleWorker  
from podsaad import models
import logging

logger = logging.getLogger(__name__)

listen = ["default"]


class PodsaadWorker(SimpleWorker):
    def __init__(self, *args, **kwargs):
        settings = kwargs.pop("settings")
        super().__init__(*args, **kwargs)
        models.init_mongoengine(settings)


class WorkerServer:
    def __init__(self, settings):
        self.settings = settings
        redis_url = settings.get("REDIS_URL", "redis://localhost:6379")
        self.conn = redis.from_url(redis_url)

    def run(self):
        queues = [Queue(name, connection=self.conn) for name in listen]
        worker = PodsaadWorker(queues, connection=self.conn, settings=self.settings)
        worker.work()