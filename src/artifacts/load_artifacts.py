import logging
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger()


def load_artifacts(artifacts_path: str) -> Tuple[BaseEstimator, StandardScaler]:
    try:
        logger.info(f"Loading model and scaler from {artifacts_path}")
        artifacts_path = Path(artifacts_path)

        model = joblib.load(artifacts_path / "model.pkl")
        scaler = joblib.load(artifacts_path / "scaler.pkl")
        train_data = pd.read_csv(artifacts_path / "train_data.csv")

        logger.info("Model and scaler loaded successfully.")
        return model, scaler, train_data

    except Exception as e:
        logger.exception("Failed to load model or scaler.")
        raise RuntimeError("Error loading model or scaler.") from e
