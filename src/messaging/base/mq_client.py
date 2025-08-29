from abc import ABC, abstractmethod


class MqClient(ABC):
    @abstractmethod
    def connect(self, max_retries=10, delay=5):
        pass

    @abstractmethod
    def send_message(self, message: dict):
        pass

    @abstractmethod
    def start_consuming(self, callback):
        pass

    @abstractmethod
    def close(self):
        pass
