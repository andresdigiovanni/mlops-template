from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional


class ExperimentTracker(ABC):
    # -----------------------------
    # Run management
    # -----------------------------

    def __enter__(self):
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run()

    @abstractmethod
    def start_run(
        self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None
    ):
        pass

    @abstractmethod
    def end_run(self):
        pass

    # -----------------------------
    # Logging
    # -----------------------------

    @abstractmethod
    def log_param(self, key: str, value: Any):
        pass

    @abstractmethod
    def log_params(self, params: dict):
        pass

    @abstractmethod
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        pass

    @abstractmethod
    def log_artifact(self, file_path: Path, category: str = None):
        pass

    # -----------------------------
    # Model handling
    # -----------------------------

    @abstractmethod
    def save_model(self, model, name: str, **kwargs):
        pass

    @abstractmethod
    def load_model(self, model_name: str, alias: str = "production"):
        pass

    @abstractmethod
    def get_artifact(self, model_name: str, path: str, alias: str = "production"):
        pass
