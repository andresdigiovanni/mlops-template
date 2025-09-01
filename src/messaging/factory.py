import logging
from typing import Any, Dict, Literal, Optional

from src.messaging.base import MqClient
from src.messaging.clients import DummyMQClient, RabbitMQClient

logger = logging.getLogger()

MqType = Literal["dummy", "rabbitmq"]


def create_messaging_client(
    mq_type: MqType, params: Optional[Dict[str, Any]] = None
) -> MqClient:
    try:
        logger.info(f"Creating messaging client: '{mq_type}'")
        params = params or {}

        if mq_type == "dummy":
            return DummyMQClient(**params)

        elif mq_type == "rabbitmq":
            return RabbitMQClient(**params)

        else:
            raise ValueError(f"Unsupported messaging client type: {mq_type}")

    except Exception as e:
        logger.exception("Error creating messaging client instance.")
        raise e
