import logging

from src.tuning import BaseModelTuner
from src.tuning.models import LightGBMTuner, LogisticRegressionTuner

logger = logging.getLogger()


def create_tuner(model_name: str, X, y, **kwargs) -> BaseModelTuner:
    try:
        if model_name == "logistic_regression":
            return LogisticRegressionTuner(X, y, **kwargs)
        elif model_name == "lightgbm":
            return LightGBMTuner(X, y, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_name}")

    except Exception as e:
        logger.exception("Error creating model tunner.")
        raise RuntimeError(f"Model creation failed: {str(e)}")
