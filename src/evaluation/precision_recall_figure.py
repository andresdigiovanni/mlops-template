import logging
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve

logger = logging.getLogger()


def plot_precision_recall_curve(
    y_true,
    y_proba,
    title: str = "Precision-Recall Curve",
    base_path: str = "artifacts",
):
    try:
        run_path = Path(base_path)
        run_path.mkdir(parents=True, exist_ok=True)

        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f"AP = {avg_precision:.2f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(title)
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(run_path / "pr_curve.png")
        plt.close()

        logger.info("Precision-Recall curve saved.")

    except Exception as e:
        logger.exception("Failed to generate Precision-Recall curve.")
        raise RuntimeError(f"Failed to generate PR curve: {str(e)}")
