import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger()


def preprocess_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Scales the data and performs a train/test split.

    Args:
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
            - scaler (StandardScaler): Fitted scaler object.
    """
    try:
        logger.info("Starting preprocessing: train/test split and scaling.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_scaled = pd.DataFrame(
            X_train_scaled, columns=X.columns, index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            X_test_scaled, columns=X.columns, index=X_test.index
        )

        logger.info(
            f"Preprocessing completed. Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}"
        )
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler

    except Exception as e:
        logger.exception("Error during preprocessing.")
        raise RuntimeError(f"Failed during preprocessing: {str(e)}")
