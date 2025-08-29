import json

import hydra
import pandas as pd

from src.experiment_tracker import create_experiment_tracker
from src.messaging import create_messaging_client
from src.monitoring import DriftDetector, DriftState


class DriftConsumer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = cfg["model"]["name"]
        self.drift_output_path = cfg["drift"]["path"]

        # Cargar artefactos
        self.tracker = create_experiment_tracker(
            cfg["experiment_tracker"]["type"], cfg["experiment_tracker"]["params"]
        )
        self._load_training_data()

        # Inicializar detector de drift
        self.drift_state = DriftState(
            cfg["drift"]["path"], buffer_size=cfg["drift"]["buffer_size"]
        )
        self.drift_detector = DriftDetector(self.training_data, self.training_preds)

        # Create messaging client
        self.messaging_client = create_messaging_client(
            cfg["messaging"]["type"], cfg["messaging"]["params"]
        )
        self.messaging_client.connect()
        self.messaging_client.start_consuming(self._callback)

    def _load_training_data(self):
        train_data_path = self.tracker.get_artifact(
            self.model_name, path="dataset/train_data.csv"
        )
        train_data = pd.read_csv(train_data_path)

        self.training_data = train_data.drop(["target", "pred", "proba"], axis=1)
        self.training_preds = train_data[["proba"]]

    def _callback(self, ch, method, properties, body):
        """Procesa cada mensaje de la cola."""
        message = json.loads(body)

        input_data = pd.read_json(message["input_data"])
        probs = pd.DataFrame(message["probabilities"], columns=["class_0", "class_1"])

        should_check = self.drift_state.add_input(input_data, probs.iloc[:, 1])

        if should_check:
            current_data, current_preds = self.drift_state.get_buffer_data()
            result = self.drift_detector.check_drift(current_data, current_preds)

            if result.get("error"):
                print(f"Drift detection failed: {result['error']}")
            else:
                print("Drift detection report generated.")
                self.drift_detector.save_report_html(self.drift_output_path)


@hydra.main(config_path="../../config", config_name="config")
def main(cfg):
    DriftConsumer(cfg)


if __name__ == "__main__":
    main()
