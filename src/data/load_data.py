import logging
from typing import Tuple

import pandas as pd
from sklearn.datasets import load_breast_cancer

logger = logging.getLogger()


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads the breast cancer dataset from sklearn and returns features and labels.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features (X) and target (y).
    """
    try:
        logger.info("Loading breast cancer dataset from sklearn...")
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")
        logger.info(f"Dataset loaded successfully with shape: X={X.shape}, y={y.shape}")
        return X, y

    except Exception as e:
        logger.exception("Error while loading the dataset.")
        raise RuntimeError(f"Failed to load dataset: {str(e)}")
