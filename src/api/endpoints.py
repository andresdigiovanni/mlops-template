import logging
from typing import List

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from hydra import compose, initialize

from src.experiment_tracker import create_experiment_tracker
from src.monitoring import DriftDetector, DriftState
from src.pipeline import run_inference_pipeline
from src.schemas import PredictionRequest, PredictionResponse

app = FastAPI(title="Breast Cancer Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LABELS = {0: "malignant", 1: "benign"}
logger = logging.getLogger()

with initialize(config_path="../../config"):
    cfg = compose(config_name="config")


model_name = cfg["model"]["name"]

# Load artifacts
tracker = create_experiment_tracker(
    cfg["experiment_tracker"]["type"], cfg["experiment_tracker"]["params"]
)

model = tracker.load_model(model_name)

scaler_path = tracker.get_artifact(model_name, artifact_path="artifact/scaler.pkl")
scaler = joblib.load(scaler_path)

train_data_path = tracker.get_artifact(
    model_name, artifact_path="dataset/train_data.csv"
)
train_data = pd.read_csv(train_data_path)

training_data = train_data.drop(["target", "pred", "proba"], axis=1)
training_preds = train_data[["proba"]]

# Initialize drift detector
drift_state = DriftState(cfg["drift"]["path"], buffer_size=cfg["drift"]["buffer_size"])
drift_detector = DriftDetector(training_data, training_preds)


@app.post("/predict", response_model=List[PredictionResponse])
def predict(inputs: List[PredictionRequest]) -> List[PredictionResponse]:
    input_df = pd.DataFrame([i.model_dump() for i in inputs])

    try:
        preds, probs = run_inference_pipeline(
            model,
            scaler,
            input_df,
            drift_detector,
            drift_state,
            cfg["drift"]["path"],
        )

        responses = []
        for pred, prob in zip(preds, probs):
            predicted_class = int(pred)
            predicted_prob = float(prob[predicted_class])
            label = LABELS[predicted_class]

            responses.append(
                PredictionResponse(
                    prediction=predicted_class,
                    prediction_prob=predicted_prob,
                    label=label,
                )
            )

        return responses

    except Exception:
        logger.exception("Prediction failed")
        return []


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
