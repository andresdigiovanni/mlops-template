import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from beaverfe import BeaverPipeline, auto_feature_pipeline
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split

logger = logging.getLogger()


def preprocess_data(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, TransformerMixin, Dict]:
    """
    Scales the data and performs a train/test split.

    Args:
        model: Model instance.
        X (pd.DataFrame): Features.
        y (pd.Series): Target labels.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random state for reproducibility.

    Returns:
        Tuple containing:
            - X_train (np.ndarray)
            - X_test (np.ndarray)
            - y_train (np.ndarray)
            - y_test (np.ndarray)
            - transformer (TransformerMixin): Fitted transformer object.
            - transformations (Dict): Transformations dict.
    """
    try:
        logger.info("Starting preprocessing: train/test split and scaling.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        transformations = auto_feature_pipeline(X_train, y_train, model, scoring)

        bfe = BeaverPipeline(transformations)
        X_train = bfe.fit_transform(X_train, y_train)
        X_test = bfe.transform(X_test, y_test)

        logger.info(
            f"Preprocessing completed. Train shape: {X_train.shape}, Test shape: {X_test.shape}"
        )
        return X_train, X_test, y_train, y_test, bfe, transformations

    except Exception as e:
        logger.exception("Error during preprocessing.")
        raise RuntimeError(f"Failed during preprocessing: {str(e)}")
