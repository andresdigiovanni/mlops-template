import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import shap

logger = logging.getLogger()


def explain_model(model: Any, X: pd.DataFrame, base_path: str = "artifacts"):
    try:
        run_path = Path(base_path)
        run_path.mkdir(parents=True, exist_ok=True)

        explainer = shap.LinearExplainer(model, X)
        shap_values = explainer.shap_values(X)

        fig = plt.figure()
        shap.summary_plot(shap_values, X, show=False)
        plt.tight_layout()
        plt.savefig(run_path / "shap_summary.png")
        plt.close(fig)

        logger.info("Generated SHAP explainer.")

    except Exception as e:
        logger.exception("SHAP explainer failed.")
        raise RuntimeError(f"Failed to explain model: {str(e)}")
