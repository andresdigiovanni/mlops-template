import random

import hydra
import joblib
import pandas as pd
from omegaconf import DictConfig

from src.experiment_tracker import create_experiment_tracker
from src.messaging import create_messaging_client
from src.pipeline import run_inference_pipeline
from src.schemas import PredictionRequest, PredictionResponse


@hydra.main(config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    LABELS = {0: "no", 1: "yes"}

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

    # Create an example of input data
    example_input = generate_random_prediction()
    input_df = pd.DataFrame([example_input.model_dump()])

    # Execute prediction
    preds, probs = run_inference_pipeline(
        model, transformer, input_df, messaging_client
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

    messaging_client.close()


def generate_random_prediction():
    JOBS = [
        "management",
        "blue-collar",
        "unemployed",
        "housemaid",
        "technician",
        "retired",
        "admin.",
        "services",
        "self-employed",
    ]
    MARITALS = ["married", "single", "divorced"]
    EDUCATIONS = ["primary", "secondary", "tertiary", "unknown"]
    YES_NO = ["yes", "no"]
    CONTACTS = ["cellular", "telephone"]
    MONTHS = [
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ]
    POUTCOMES = ["failure", "success", "other", "unknown"]

    return PredictionRequest(
        age=random.randint(18, 90),
        job=random.choice(JOBS),
        marital=random.choice(MARITALS),
        education=random.choice(EDUCATIONS),
        default=random.choice(YES_NO),
        balance=randomize_value(1500, percent=1.0),
        housing=random.choice(YES_NO),
        loan=random.choice(YES_NO),
        contact=random.choice(CONTACTS),
        day=random.randint(1, 31),
        month=random.choice(MONTHS),
        duration=randomize_value(300, percent=1.0),
        campaign=random.randint(1, 10),
        pdays=randomize_value(50, percent=1.0),
        previous=random.randint(0, 10),
        poutcome=random.choice(POUTCOMES),
    )


def randomize_value(base, percent=0.1):
    delta = base * percent
    return round(random.uniform(base - delta, base + delta), 4)


if __name__ == "__main__":
    main()
