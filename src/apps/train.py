from pathlib import Path
from time import sleep

from src.pipeline import run_training_pipeline

if __name__ == "__main__":
    config_file = "config_lr.yaml"
    config_path = Path("config") / config_file
    run_training_pipeline(config_path)
