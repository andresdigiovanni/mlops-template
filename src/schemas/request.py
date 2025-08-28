from pydantic import BaseModel


class PredictionRequest(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: float
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    duration: float
    campaign: int
    pdays: float
    previous: int
    poutcome: str

    class Config:
        schema_extra = {
            "example": {
                "age": 48,
                "job": "management",
                "marital": "married",
                "education": "tertiary",
                "default": "no",
                "balance": 0.0,
                "housing": "yes",
                "loan": "no",
                "contact": "cellular",
                "day": 4,
                "month": "may",
                "duration": 85.0,
                "campaign": 1,
                "pdays": 61.0,
                "previous": 5,
                "poutcome": "failure",
            }
        }
