import logging
from typing import Any

logger = logging.getLogger()


def train_model(model: Any, X_train, y_train) -> Any:
    """
    Trains a given model with training data.

    Args:
        model: Untrained model instance.
        X_train: Features for training.
        y_train: Labels for training.

    Returns:
        Trained model instance.
    """
    try:
        logger.info("Starting model training...")
        model.fit(X_train, y_train)
        logger.info("Model training completed.")
        return model

    except Exception as e:
        logger.exception("Error during training.")
        raise RuntimeError(f"Training failed: {str(e)}")
