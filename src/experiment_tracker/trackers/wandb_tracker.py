from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import wandb

from src.experiment_tracker import ExperimentTracker


class WandbTracker(ExperimentTracker):
    def __init__(self, experiment_name: str, entity: Optional[str] = None):
        wandb.login()
        self.experiment_name = experiment_name
        self.entity = entity

        self.base_dir = Path("wandb_artifacts")
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.run = None
        self.run_id = None

        # Buffer local
        self._artifact_files = []  # (file_path, artifact_path)
        self._model_path = None  # ruta del modelo guardado
        self._registered_model_name = None

    # -----------------------------
    # Run management
    # -----------------------------
    def start_run(
        self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None
    ):
        self.run = wandb.init(
            project=self.experiment_name,
            entity=self.entity,
            name=run_name,
            config=tags or {},
        )
        self.run_id = self.run.id
        return self

    def end_run(self):
        if not self.run:
            return

        # Crear único artifact con todo dentro
        artifact = wandb.Artifact(self._registered_model_name, type="model")

        # Modelo
        if self._model_path:
            artifact.add_file(str(self._model_path), name="model.pkl")

        # Otros artefactos con estructura de carpetas
        for file_path, art_path in self._artifact_files:
            rel_name = (
                f"{art_path}/{Path(file_path).name}"
                if art_path
                else Path(file_path).name
            )
            artifact.add_file(str(file_path), name=rel_name)

        # Log con alias "production"
        self.run.log_artifact(artifact, aliases=["production"])

        self.run.finish()
        self.run = None
        self._artifact_files.clear()
        self._model_path = None

    # -----------------------------
    # Logging
    # -----------------------------
    def log_param(self, key: str, value: Any):
        self.run.config[key] = value

    def log_params(self, params: dict):
        self.run.config.update(params)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        self.run.log({key: value}, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        self.run.log(metrics, step=step)

    def log_artifact(self, file_path: Path, category: str = None):
        if category == "images":
            self.run.log({file_path.stem: wandb.Image(file_path)})
        else:
            self._artifact_files.append((file_path, category))

    # -----------------------------
    # Model handling
    # -----------------------------
    def save_model(
        self,
        model,
        name: str,
        registered_model_name: str = None,  # ignorado pero mantenido por coherencia
        input_example=None,
        output_example=None,
    ):
        model_path = self.base_dir / "model.pkl"
        joblib.dump(model, model_path)
        self._model_path = model_path
        self._registered_model_name = registered_model_name or name

    def load_model(self, model_name: str, alias: str = "production"):
        api = wandb.Api()
        artifact_ref = f"{self.experiment_name}/{model_name}:{alias}"
        if self.entity:
            artifact_ref = f"{self.entity}/{artifact_ref}"

        artifact = api.artifact(artifact_ref, type="model")
        model_dir = Path(artifact.download())
        return joblib.load(model_dir / "model.pkl")

    def get_artifact(self, model_name: str, path: str, alias: str = "production"):
        api = wandb.Api()
        artifact_ref = f"{self.experiment_name}/{model_name}:{alias}"
        if self.entity:
            artifact_ref = f"{self.entity}/{artifact_ref}"

        artifact = api.artifact(artifact_ref, type="model")
        artifact_dir = Path(artifact.download())
        target_path = artifact_dir / path

        if not target_path.exists():
            raise FileNotFoundError(f"Artifact {path} not found in {artifact_dir}")

        return str(target_path)
