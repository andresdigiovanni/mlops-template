import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

logger = logging.getLogger()


def plot_confusion_matrix(
    y_true,
    y_pred,
    normalize=False,
    title="Confusion Matrix",
    base_path: str = "artifacts",
):
    try:
        run_path = Path(base_path)
        run_path.mkdir(parents=True, exist_ok=True)

        labels = np.unique(np.concatenate([y_true, y_pred]))

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
            normalize="true" if normalize else None,
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(cax, ax=ax)

        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:.2f}" if normalize else f"{cm[i, j]:d}",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        ax.set_title(title)

        plt.tight_layout()
        plt.savefig(run_path / "confusion_matrix.png")
        plt.close(fig)

        logger.info("Generated confusion matrix figure.")

    except Exception as e:
        logger.exception("Generate confusion matrix failed.")
        raise RuntimeError(f"Failed to generate confusion matrix figure: {str(e)}")
