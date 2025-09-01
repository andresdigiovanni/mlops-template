import logging

from src.messaging.base import MqClient

logger = logging.getLogger()


class DummyMQClient(MqClient):
    def connect(self):
        pass

    def send_message(self, message: dict):
        pass

    def start_consuming(self, callback):
        pass

    def close(self):
        pass
