import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.features import preprocess_data
from src.messaging import MqClient

logger = logging.getLogger()


def run_inference_pipeline(
    model: BaseEstimator,
    transformer: TransformerMixin,
    X: pd.DataFrame,
    messaging_client: MqClient,
) -> np.ndarray:
    try:
        logger.info(f"Running prediction on input shape: {X.shape}")
        X = preprocess_data(X)
        X = transformer.transform(X)

        preds = model.predict(X)
        probs = model.predict_proba(X)

        logger.info("Prediction completed.")

        _send_mq_message(messaging_client, X.to_json(orient="records"), probs.tolist())

        return preds, probs

    except Exception as e:
        logger.exception("Error during prediction.")
        raise RuntimeError("Prediction failed.") from e


def _send_mq_message(messaging_client: MqClient, input_data, probs):
    message = {
        "input_data": input_data,
        "probabilities": probs,
    }
    messaging_client.send_message(message)
