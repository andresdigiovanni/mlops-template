import logging
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

logger = logging.getLogger()


def plot_roc_curve(
    y_true,
    y_proba,
    title: str = "ROC Curve",
    base_path: str = "artifacts",
):
    try:
        run_path = Path(base_path)
        run_path.mkdir(parents=True, exist_ok=True)

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(run_path / "roc_curve.png")
        plt.close()

        logger.info("ROC curve saved.")

    except Exception as e:
        logger.exception("Failed to generate ROC curve.")
        raise RuntimeError(f"Failed to generate ROC curve: {str(e)}")
