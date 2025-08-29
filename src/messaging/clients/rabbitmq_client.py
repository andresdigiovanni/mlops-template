import json
import logging
import time

import pika

from src.messaging.base import MqClient

logger = logging.getLogger()


class RabbitMQClient(MqClient):
    def __init__(self, host: str, port: int, queue: str):
        self.host = host
        self.port = port
        self.queue = queue
        self.connection = None
        self.channel = None

    def connect(self, max_retries=10, delay=5):
        """Conectar con RabbitMQ con reintentos."""
        for i in range(max_retries):
            try:
                self.connection = pika.BlockingConnection(
                    pika.ConnectionParameters(host=self.host, port=self.port)
                )
                self.channel = self.connection.channel()
                self.channel.queue_declare(queue=self.queue)
                logger.info("Conectado a RabbitMQ")
                return

            except pika.exceptions.AMQPConnectionError:
                logger.warning(
                    f"RabbitMQ no disponible, reintento {i + 1}/{max_retries} en {delay}s..."
                )
                time.sleep(delay)

        raise ConnectionError(
            "No se pudo conectar a RabbitMQ después de varios intentos."
        )

    def send_message(self, message: dict):
        """Publicar un mensaje en la cola."""
        if self.channel is None:
            raise ConnectionError("No conectado a RabbitMQ")

        self.channel.basic_publish(
            exchange="",
            routing_key=self.queue,
            body=json.dumps(message),
        )
        logger.info(f"Mensaje enviado a RabbitMQ: {message}")

    def start_consuming(self, callback):
        """Comenzar a consumir mensajes de la cola."""
        if self.channel is None:
            raise ConnectionError("No conectado a RabbitMQ")

        self.channel.basic_consume(
            queue=self.queue,
            on_message_callback=callback,
            auto_ack=True,
        )
        logger.info("Esperando mensajes...")
        self.channel.start_consuming()

    def close(self):
        """Cerrar conexión."""
        if self.connection and self.connection.is_open:
            self.connection.close()
            logger.info("Conexión a RabbitMQ cerrada")
