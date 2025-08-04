from threading import Thread

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from src.monitoring import DriftDetector, DriftState
from src.utils import get_logger

logger = get_logger()


def run_inference_pipeline(
    model: BaseEstimator,
    scaler: StandardScaler,
    input_data: pd.DataFrame,
    drift_detector: DriftDetector,
    drift_state: DriftState,
    drift_output_path: str,
) -> np.ndarray:
    try:
        logger.info(f"Running prediction on input shape: {input_data.shape}")
        X_scaled = scaler.transform(input_data)

        preds = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)

        logger.info("Prediction completed.")

        # Lazy init DriftDetector
        should_check = drift_state.add_input(input_data, probs[:, 1])

        if should_check:
            logger.info("Drift buffer full. Launching async drift detection...")
            Thread(
                target=_run_drift_check_async,
                args=(drift_detector, drift_state, drift_output_path),
            ).start()

        return preds, probs

    except Exception as e:
        logger.exception("Error during prediction.")
        raise RuntimeError("Prediction failed.") from e


def _run_drift_check_async(
    drift_detector: DriftDetector, drift_state: DriftState, output_path: str
):
    """
    Asynchronous drift check and report generation.
    """
    try:
        current_data, current_preds = drift_state.get_buffer_data()
        result = drift_detector.check_drift(current_data, current_preds)

        if result.get("error"):
            logger.error(f"Drift detection failed: {result['error']}")
        else:
            logger.info("Drift detection generated.")
            drift_detector.save_report_html(output_path)

    except Exception:
        logger.exception("Error during async drift detection.")
