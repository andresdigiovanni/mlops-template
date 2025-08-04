from src.artifacts import save_artifacts
from src.data import load_data, normalize_column_names
from src.evaluation import evaluate_model
from src.features import preprocess_data
from src.models import create_model, train_model
from src.utils import get_logger, load_config


def run_training_pipeline(config_path: str = "config.yaml") -> None:
    logger = get_logger()
    logger.info("Starting ML training pipeline")

    try:
        # 1. Load config
        config = load_config(config_path)

        # 2. Load data
        X, y = load_data()
        X = normalize_column_names(X)

        # 3. Preprocess
        X_train, X_test, y_train, y_test, scaler = preprocess_data(
            X,
            y,
            test_size=config["data"]["test_size"],
            random_state=config["data"]["random_state"],
        )

        # 4. Model creation
        model = create_model(
            model_type=config["model"]["type"], params=config["model"].get("params")
        )

        # 5. Train
        model = train_model(model, X_train, y_train)

        # 6. Evaluate
        metrics, cm = evaluate_model(model, X_test, y_test)

        # 7. Training data
        y_pred_train = model.predict(X_train)
        y_proba_train = (
            model.predict_proba(X_train)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        train_data_snapshot = X_train.copy()
        train_data_snapshot["target"] = y_train
        train_data_snapshot["pred"] = y_pred_train
        train_data_snapshot["proba"] = y_proba_train

        # 8. Save
        save_artifacts(
            model=model,
            scaler=scaler,
            train_data=train_data_snapshot,
            metrics=metrics,
            config=config,
            confusion_matrix=cm,
            base_path=config["artifacts"]["path"],
        )

        logger.info("Training pipeline completed.")

    except Exception as e:
        logger.exception("Error in training pipeline.")
        raise e
