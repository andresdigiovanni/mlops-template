import logging
from typing import List

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from hydra import compose, initialize

from src.experiment_tracker import create_experiment_tracker
from src.messaging import create_messaging_client
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

transformer_path = tracker.get_artifact(model_name, path="artifact/transformer.pkl")
transformer = joblib.load(transformer_path)

# Create messaging client
messaging_client = create_messaging_client(
    cfg["messaging"]["type"], cfg["messaging"]["params"]
)
messaging_client.connect()


@app.post("/predict", response_model=List[PredictionResponse])
def predict(inputs: List[PredictionRequest]) -> List[PredictionResponse]:
    input_df = pd.DataFrame([i.model_dump() for i in inputs])

    try:
        preds, probs = run_inference_pipeline(
            model, transformer, input_df, messaging_client
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
