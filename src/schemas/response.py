from typing import List

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    prediction: int  # 0 or 1 for binary classification
    prediction_prob: float  # [0, 1] for probability classification
    label: str  # 'benign' or 'malignant'
