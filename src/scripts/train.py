import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from src.pipeline import run_training_pipeline


@hydra.main(config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    load_dotenv()

    run_training_pipeline(cfg)


if __name__ == "__main__":
    main()
