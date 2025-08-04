from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.utils import get_logger

logger = get_logger()


class DriftDetector:
    def __init__(self, training_data, training_preds):
        self.training_data = training_data
        self.training_preds = training_preds
        self.last_data_drift_report: Report | None = None
        self.last_pred_drift_report: Report | None = None

    def save_report_html(self, path: str):
        if self.last_data_drift_report is None or self.last_pred_drift_report is None:
            logger.info("No drift report generated")
            return

        try:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)

            self.last_data_drift_report.save_html(str(path / "data_drift.html"))
            self.last_pred_drift_report.save_html(str(path / "pred_drift.html"))

            logger.info(f"Drift report saved to {path}")
        except Exception:
            logger.exception("Failed to save drift report.")
            raise

    def check_drift(self, current_data: pd.DataFrame, current_preds: pd.DataFrame):
        try:
            data_drift_json, is_data_drift_detected, self.last_data_drift_report = (
                self._execute_check_drift(current_data, self.training_data)
            )
            pred_drift_json, is_pred_drift_detected, self.last_pred_drift_report = (
                self._execute_check_drift(current_preds, self.training_preds)
            )

            return {
                "data_drift": data_drift_json,
                "is_data_drift_detected": is_data_drift_detected,
                "prediction_drift": pred_drift_json,
                "is_pred_drift_detected": is_pred_drift_detected,
            }
        except Exception as e:
            logger.exception("Failed to generate drift report.")
            return {"error": str(e)}

    def _execute_check_drift(self, current_data, reference_data):
        report = Report([DataDriftPreset()], include_tests=True)
        result = report.run(current_data, reference_data)

        data_drift_json = result.dict()
        is_data_drift_detected = any(
            test["status"].value == "FAIL" for test in data_drift_json.get("tests", [])
        )

        return data_drift_json, is_data_drift_detected, result
