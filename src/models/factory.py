import logging
from typing import Any, Dict, Literal, Optional

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger()

ModelType = Literal["logistic_regression", "lightgbm"]


def create_model(model_type: ModelType, params: Optional[Dict[str, Any]] = None) -> Any:
    """
    Factory function to create a model based on the given type and parameters.

    Args:
        model_type (str): Type of model to create ('logistic_regression' or 'lightgbm').
        params (dict, optional): Hyperparameters for the model.

    Returns:
        Untrained model instance.

    Raises:
        ValueError: If model_type is invalid.
    """
    try:
        logger.info(
            f"Creating model of type: '{model_type}' with params: {params or 'default'}"
        )
        params = params or {}

        if model_type == "logistic_regression":
            return LogisticRegression(**params)
        elif model_type == "lightgbm":
            return lgb.LGBMClassifier(**params, verbose=-1)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    except Exception as e:
        logger.exception("Error creating model instance.")
        raise RuntimeError(f"Model creation failed: {str(e)}")
