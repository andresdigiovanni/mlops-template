import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger()


def save_artifacts(
    model: Any,
    params: Dict,
    scaler: Any,
    train_data: pd.DataFrame,
    metrics: Dict[str, Any],
    confusion_matrix: Optional[np.ndarray] = None,
    base_path: str = "artifacts",
) -> Path:
    """
    Saves model, params, scaler, metrics, and confusion matrix to a timestamped artifact directory.

    Args:
        model: Trained model instance.
        params: Model parameters.
        scaler: Fitted scaler instance.
        metrics (dict): Evaluation metrics.
        confusion_matrix (np.ndarray, optional): Confusion matrix to save as heatmap.
        base_path (str): Base directory for saving artifacts.

    Returns:
        Path: Path to the created artifact directory.
    """
    try:
        run_path = Path(base_path)
        run_path.mkdir(parents=True, exist_ok=True)

        # Save model
        joblib.dump(model, run_path / "model.pkl")
        logger.info(f"Model saved to {run_path / 'model.pkl'}")

        # Save parameters
        with open(run_path / "params.json", "w") as f:
            json.dump(params, f, indent=4)
        logger.info(f"Params saved to {run_path / 'params.json'}")

        # Save scaler
        joblib.dump(scaler, run_path / "scaler.pkl")
        logger.info(f"Scaler saved to {run_path / 'scaler.pkl'}")

        # Save training data
        train_data.to_csv(run_path / "train_data.csv", index=False)

        # Save metrics
        with open(run_path / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to {run_path / 'metrics.json'}")

        # Save confusion matrix plot if provided
        if confusion_matrix is not None:
            plt.figure(figsize=(6, 5))
            sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            plt.savefig(run_path / "confusion_matrix.png")
            plt.close()
            logger.info(
                f"Confusion matrix plot saved to {run_path / 'confusion_matrix.png'}"
            )

        return run_path

    except Exception as e:
        logger.exception("Error saving artifacts.")
        raise RuntimeError(f"Failed to save artifacts: {str(e)}")
