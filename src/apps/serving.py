import random

import hydra
import joblib
import pandas as pd
from omegaconf import DictConfig

from src.experiment_tracker import create_experiment_tracker
from src.monitoring import DriftDetector, DriftState
from src.pipeline import run_inference_pipeline
from src.schemas import PredictionRequest, PredictionResponse


@hydra.main(config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    LABELS = {0: "malignant", 1: "benign"}

    model_name = cfg["model"]["name"]

    # Load artifacts
    tracker = create_experiment_tracker(
        cfg["experiment_tracker"]["type"], cfg["experiment_tracker"]["params"]
    )

    model = tracker.load_model(model_name)

    transformer_path = tracker.get_artifact(model_name, path="artifact/transformer.pkl")
    transformer = joblib.load(transformer_path)

    train_data_path = tracker.get_artifact(model_name, path="dataset/train_data.csv")
    train_data = pd.read_csv(train_data_path)

    training_data = train_data.drop(["target", "pred", "proba"], axis=1)
    training_preds = train_data[["proba"]]

    # Initialize drift detector
    drift_state = DriftState(
        cfg["drift"]["path"], buffer_size=cfg["drift"]["buffer_size"]
    )
    drift_detector = DriftDetector(training_data, training_preds)

    # Create an example of input data
    example_input = generate_random_prediction()
    input_df = pd.DataFrame([example_input.model_dump()])

    # Execute prediction
    preds, probs = run_inference_pipeline(
        model, transformer, input_df, drift_detector, drift_state, cfg["drift"]["path"]
    )
    responses = []

    for pred, prob in zip(preds, probs):
        predicted_class = int(pred)
        predicted_prob = float(prob[predicted_class])
        label = LABELS[predicted_class]

        response = PredictionResponse(
            prediction=predicted_class, prediction_prob=predicted_prob, label=label
        )
        responses.append(response)

    # Show result
    for r in responses:
        print(r.model_dump_json())


def generate_random_prediction():
    return PredictionRequest(
        mean_radius=randomize_value(14.5),
        mean_texture=randomize_value(20.5),
        mean_perimeter=randomize_value(96.2),
        mean_area=randomize_value(644.1),
        mean_smoothness=randomize_value(0.105),
        mean_compactness=randomize_value(0.15),
        mean_concavity=randomize_value(0.12),
        mean_concave_points=randomize_value(0.075),
        mean_symmetry=randomize_value(0.18),
        mean_fractal_dimension=randomize_value(0.06),
        radius_error=randomize_value(0.55),
        texture_error=randomize_value(1.0),
        perimeter_error=randomize_value(3.3),
        area_error=randomize_value(40.0),
        smoothness_error=randomize_value(0.006),
        compactness_error=randomize_value(0.015),
        concavity_error=randomize_value(0.02),
        concave_points_error=randomize_value(0.01),
        symmetry_error=randomize_value(0.015),
        fractal_dimension_error=randomize_value(0.003),
        worst_radius=randomize_value(16.5),
        worst_texture=randomize_value(28.0),
        worst_perimeter=randomize_value(105.0),
        worst_area=randomize_value(800.0),
        worst_smoothness=randomize_value(0.13),
        worst_compactness=randomize_value(0.2),
        worst_concavity=randomize_value(0.25),
        worst_concave_points=randomize_value(0.15),
        worst_symmetry=randomize_value(0.25),
        worst_fractal_dimension=randomize_value(0.08),
    )


def randomize_value(base, percent=0.1):
    delta = base * percent
    return round(random.uniform(base - delta, base + delta), 4)


if __name__ == "__main__":
    main()
