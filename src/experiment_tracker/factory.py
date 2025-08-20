import logging
from typing import Any, Dict, Literal, Optional

from src.experiment_tracker import ExperimentTracker
from src.experiment_tracker.trackers import LocalTracker, MLflowTracker, WandbTracker

logger = logging.getLogger()

TrackerType = Literal["local", "mlflow", "wandb"]


def create_experiment_tracker(
    tracker_type: TrackerType, params: Optional[Dict[str, Any]] = None
) -> ExperimentTracker:
    try:
        logger.info(f"Creating experiment tracker of type: '{tracker_type}'")
        params = params or {}

        if tracker_type == "local":
            return LocalTracker(**params)

        elif tracker_type == "mlflow":
            return MLflowTracker(**params)

        elif tracker_type == "wandb":
            return WandbTracker(**params)

        else:
            raise ValueError(f"Unsupported experiment tracker type: {tracker_type}")

    except Exception as e:
        logger.exception("Error creating experiment tracker instance.")
        raise e
