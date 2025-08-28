from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

from src.experiment_tracker import ExperimentTracker


class MLflowTracker(ExperimentTracker):
    def __init__(self, tracking_uri: str, experiment_name: str):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()
        self.run_id = None

    # -----------------------------
    # Run management
    # -----------------------------

    def start_run(
        self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None
    ):
        run = mlflow.start_run(run_name=run_name, tags=tags)
        self.run_id = run.info.run_id
        return run

    def end_run(self):
        mlflow.end_run()

    # -----------------------------
    # Logging
    # -----------------------------

    def log_param(self, key: str, value: Any):
        mlflow.log_param(key, value)

    def log_params(self, params: dict):
        mlflow.log_params(params)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        mlflow.log_metric(key, value, step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        mlflow.log_metrics(metrics, step)

    def log_artifact(self, file_path: Path, category: str = None):
        mlflow.log_artifact(file_path, artifact_path=category)

    # -----------------------------
    # Model handling
    # -----------------------------

    def save_model(
        self,
        model,
        name: str,
        registered_model_name: str = None,
        input_example=None,
        output_example=None,
    ):
        signature = infer_signature(input_example, output_example)

        mlflow.sklearn.log_model(
            sk_model=model,
            name=name,
            registered_model_name=registered_model_name,
            input_example=input_example,
            signature=signature,
        )

        if registered_model_name:
            latest_versions = self.client.get_latest_versions(registered_model_name)
            new_version = latest_versions[0].version
            self.client.set_registered_model_alias(
                registered_model_name, "production", str(new_version)
            )

    def load_model(self, model_name: str, alias: str = "production"):
        return mlflow.sklearn.load_model(f"models:/{model_name}@{alias}")

    def get_artifact(self, model_name: str, path: str, alias: str = "production"):
        run_id = self.client.get_model_version_by_alias(model_name, alias).run_id

        return mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=path)
