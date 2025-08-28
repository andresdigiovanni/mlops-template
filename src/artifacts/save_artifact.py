import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

logger = logging.getLogger()


def save_artifact(obj: Any, file_path: Path) -> Path:
    """
    Saves an object to disk using the appropriate method depending on the file extension.

    Args:
        obj: Object to save (model, dict, DataFrame, scaler, etc.).
        file_path (Path): Directory where the file will be saved.

    Returns:
        Path: Path to the saved file.
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        ext = file_path.suffix.lower()

        if ext == ".pkl":
            joblib.dump(obj, file_path)

        elif ext == ".json":
            with open(file_path, "w") as f:
                json.dump(obj, f, indent=4)

        elif ext == ".csv":
            if isinstance(obj, pd.DataFrame):
                obj.to_csv(file_path, index=False)
            else:
                raise ValueError("Only pandas DataFrames can be saved as CSV.")

        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        logger.info(f"Object saved to {file_path}")
        return file_path

    except Exception as e:
        logger.exception("Error saving object.")
        raise RuntimeError(f"Failed to save {file_path}: {str(e)}") from e
