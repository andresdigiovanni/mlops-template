import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from src.experiment_tracker import ExperimentTracker


class LocalTracker(ExperimentTracker):
    def __init__(self, base_dir: str, experiment_name: str):
        self.experiments_dir = Path(base_dir) / "experiments" / experiment_name
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        self.models_dir = Path(base_dir) / "models" / experiment_name
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.current_run_dir = None
        self.run_id = None
        self.run_data = {
            "params": {},
            "metrics": {},
            "artifacts": [],
            "tags": {},
            "start_time": None,
            "end_time": None,
        }

    # -----------------------------
    # Run management
    # -----------------------------
    def start_run(
        self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None
    ):
        self.run_id = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_run_dir = self.experiments_dir / self.run_id
        self.current_run_dir.mkdir(parents=True, exist_ok=True)

        self.run_data["start_time"] = datetime.now().isoformat()
        if tags:
            self.run_data["tags"] = tags

        return self

    def end_run(self):
        if not self.current_run_dir:
            raise RuntimeError("No active run to end.")

        self.run_data["end_time"] = datetime.now().isoformat()
        with open(self.current_run_dir / "run.json", "w") as f:
            json.dump(self.run_data, f, indent=2)

        self.current_run_dir = None

    # -----------------------------
    # Logging
    # -----------------------------
    def log_param(self, key: str, value: Any):
        self.run_data["params"][key] = value

    def log_params(self, params: dict):
        self.run_data["params"].update(params)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        if key not in self.run_data["metrics"]:
            self.run_data["metrics"][key] = []
        self.run_data["metrics"][key].append({"step": step, "value": value})

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        for k, v in metrics.items():
            self.log_metric(k, v, step)

    def log_artifact(self, file_path: Path, category: str = None):
        if not self.current_run_dir:
            raise RuntimeError("No active run. Call start_run() first.")

        artifact_dir = (
            self.current_run_dir / category if category else self.current_run_dir
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)

        target_file = artifact_dir / file_path.name
        shutil.copy(file_path, target_file)

        self.run_data["artifacts"].append(
            {
                "name": target_file.name,
                "relative_path": str(target_file.relative_to(self.experiments_dir)),
            }
        )

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
        if not self.run_id:
            raise RuntimeError("No active run. Call start_run() before save_model().")

        # Calcular siguiente versión
        existing_versions = [
            d
            for d in os.listdir(self.models_dir)
            if os.path.isdir(os.path.join(self.models_dir, d)) and d.startswith("v")
        ]
        if existing_versions:
            max_version = max(int(v[1:]) for v in existing_versions)
            new_version = f"v{max_version + 1}"
        else:
            new_version = "v1"

        version_dir = self.models_dir / new_version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Guardar modelo
        model_path = version_dir / f"{name}.pkl"
        joblib.dump(model, model_path)

        # Guardar metadatos de versiones
        versions_file = self.models_dir / "versions.json"
        if versions_file.exists():
            with open(versions_file, "r") as f:
                versions_data = json.load(f)
        else:
            versions_data = {}

        versions_data[new_version] = {
            "model_file": f"{name}.pkl",
            "run_id": self.run_id,
        }
        with open(versions_file, "w") as f:
            json.dump(versions_data, f, indent=2)

        # Si hay nombre registrado, asignar alias "production"
        if registered_model_name:
            aliases_file = self.models_dir / "aliases.json"
            if aliases_file.exists():
                with open(aliases_file, "r") as f:
                    aliases_data = json.load(f)
            else:
                aliases_data = {}
            aliases_data["production"] = new_version
            with open(aliases_file, "w") as f:
                json.dump(aliases_data, f, indent=2)

    def load_model(self, model_name: str, alias: str = "production"):
        aliases_file = self.models_dir / "aliases.json"
        if not aliases_file.exists():
            raise FileNotFoundError(f"No aliases.json found in {self.models_dir}")

        with open(aliases_file, "r") as f:
            aliases_data = json.load(f)

        if alias not in aliases_data:
            raise ValueError(f"Alias '{alias}' not found in aliases.json")

        version = aliases_data[alias]
        versions_file = self.models_dir / "versions.json"
        with open(versions_file, "r") as f:
            versions_data = json.load(f)

        if version not in versions_data:
            raise FileNotFoundError(f"Version '{version}' not found in versions.json")

        model_file = versions_data[version]["model_file"]
        model_path = self.models_dir / version / model_file

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        return joblib.load(model_path)

    def get_artifact(self, model_name: str, path: str, alias: str = "production"):
        # Resolver alias → versión
        aliases_file = self.models_dir / "aliases.json"
        with open(aliases_file, "r") as f:
            aliases_data = json.load(f)

        version = aliases_data.get(alias)
        if not version:
            raise ValueError(f"Alias '{alias}' not found for model '{model_name}'")

        # Obtener run_id desde versions.json
        versions_file = self.models_dir / "versions.json"
        with open(versions_file, "r") as f:
            versions_data = json.load(f)

        run_id = versions_data[version]["run_id"]

        # Localizar artefacto en el run correspondiente
        run_dir = self.experiments_dir / run_id
        target_path = run_dir / path
        if not target_path.exists():
            raise FileNotFoundError(f"Artifact not found at {target_path}")

        return str(target_path)
