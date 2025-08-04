from pathlib import Path

import pandas as pd

from src.artifacts import load_artifacts
from src.monitoring import DriftDetector, DriftState
from src.pipeline import run_inference_pipeline
from src.schemas import PredictionRequest, PredictionResponse
from src.utils import load_config


def main():
    LABELS = {0: "malignant", 1: "benign"}

    # 1. Load config
    config_file = "config_lr.yaml"
    config_path = Path("config") / config_file
    config = load_config(config_path)

    # 2. Load artifacts
    model, scaler, train_data = load_artifacts(config["artifacts"]["path"])
    training_data = train_data.drop(["target", "pred", "proba"], axis=1)
    training_preds = train_data[["proba"]]

    # 3. Initialize drift detector
    drift_state = DriftState(
        config["drift"]["path"], buffer_size=config["drift"]["buffer_size"]
    )
    drift_detector = DriftDetector(training_data, training_preds)

    # 4. Create an example of input data
    example_input = PredictionRequest(
        mean_radius=14.5,
        mean_texture=20.5,
        mean_perimeter=96.2,
        mean_area=644.1,
        mean_smoothness=0.105,
        mean_compactness=0.15,
        mean_concavity=0.12,
        mean_concave_points=0.075,
        mean_symmetry=0.18,
        mean_fractal_dimension=0.06,
        radius_error=0.55,
        texture_error=1.0,
        perimeter_error=3.3,
        area_error=40.0,
        smoothness_error=0.006,
        compactness_error=0.015,
        concavity_error=0.02,
        concave_points_error=0.01,
        symmetry_error=0.015,
        fractal_dimension_error=0.003,
        worst_radius=16.5,
        worst_texture=28.0,
        worst_perimeter=105.0,
        worst_area=800.0,
        worst_smoothness=0.13,
        worst_compactness=0.2,
        worst_concavity=0.25,
        worst_concave_points=0.15,
        worst_symmetry=0.25,
        worst_fractal_dimension=0.08,
    )
    input_df = pd.DataFrame([example_input.model_dump()])

    # 5. Execute prediction
    preds, probs = run_inference_pipeline(
        model, scaler, input_df, drift_detector, drift_state, config["drift"]["path"]
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

    # 6. Show result
    for r in responses:
        print(r.model_dump_json())


if __name__ == "__main__":
    main()
