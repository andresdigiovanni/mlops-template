import logging
from typing import List

import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from hydra import compose

from src.artifacts import load_artifacts
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
cfg = compose(config_path="../../config", config_name="config")

model, scaler, train_data = load_artifacts(cfg["artifacts"]["path"])
training_data = train_data.drop(columns=["target", "pred", "proba"])
training_preds = train_data[["proba"]]

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
