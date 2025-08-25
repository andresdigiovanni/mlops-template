import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import shap

logger = logging.getLogger()


def explain_model(
    model: Any, X: pd.DataFrame, explainer_type: str, base_path: str = "artifacts"
) -> None:
    try:
        output_dir = Path(base_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        explainer = _get_explainer(model, X, explainer_type)
        shap_values = explainer.shap_values(X)
        _save_summary_plot(shap_values, X, output_dir)

        logger.info(f"Generated SHAP explanation for type {explainer_type}")

    except Exception as e:
        logger.exception("SHAP explainer failed.")
        raise RuntimeError(f"Failed to explain model: {str(e)}")


def _get_explainer(model, X, explainer_type):
    if explainer_type == "linear":
        return shap.LinearExplainer(model, X)

    elif explainer_type == "tree":
        return shap.TreeExplainer(model)

    raise ValueError(f"No explainer type supported {explainer_type}")


def _save_summary_plot(shap_values, X, output_dir: Path) -> None:
    fig = plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png")
    plt.close(fig)
