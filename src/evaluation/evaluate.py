import logging
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger()


def evaluate_model(
    model: Any, X_test: np.ndarray, y_test: np.ndarray
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Evaluates a model on the test set.

    Args:
        model: Trained model instance.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.

    Returns:
        Tuple:
            - Dict[str, Any]: Dictionary with evaluation metrics.
            - np.ndarray: Confusion matrix.
    """
    try:
        logger.info("Evaluating model...")

        y_pred = model.predict(X_test)
        y_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
        }

        if y_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
            metrics["log_loss"] = log_loss(y_test, y_proba)

        logger.info("Evaluation completed.")
        return metrics, y_pred, y_proba

    except Exception as e:
        logger.exception("Model evaluation failed.")
        raise RuntimeError(f"Failed to evaluate model: {str(e)}")
