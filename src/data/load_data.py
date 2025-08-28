import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

logger = logging.getLogger()


def load_data(path: str, target: str = "target") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads dataset from the specified file path and returns features and labels.
    Supports CSV and Parquet files. Assumes the target column is named 'target'.

    Args:
        path (str): Path to the data file.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features (X) and target (y).
    """
    try:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = file_path.suffix.lower()
        logger.info(f"Loading dataset from {file_path}...")

        if ext == ".csv":
            df = pd.read_csv(file_path)

        elif ext in [".parquet", ".parq"]:
            df = pd.read_parquet(file_path)

        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        if target not in df.columns:
            raise ValueError("Dataset must contain a 'target' column")

        X = df.drop(columns=target)
        y = df[target]

        logger.info(f"Dataset loaded successfully with shape: X={X.shape}, y={y.shape}")
        return X, y

    except Exception as e:
        logger.exception("Error while loading the dataset.")
        raise RuntimeError(f"Failed to load dataset: {str(e)}")
