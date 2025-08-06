import logging

from src.artifacts import save_artifacts
from src.data import load_data, normalize_column_names
from src.evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from src.explainer import explain_model
from src.features import preprocess_data
from src.models import create_model
from src.tuning import create_tuner


def run_training_pipeline(cfg) -> None:
    logger = logging.getLogger()
    logger.info("Starting ML training pipeline")

    try:
        # Load data
        X, y = load_data()
        X = normalize_column_names(X)

        # Preprocess
        X_train, X_test, y_train, y_test, scaler = preprocess_data(
            X,
            y,
            test_size=cfg["data"]["test_size"],
            random_state=cfg["data"]["random_state"],
        )

        # Hyperparameter tuner
        tuner = create_tuner(cfg["model"]["type"], X_train, y_train)
        best_params = tuner.run()

        # Model creation
        model = create_model(model_type=cfg["model"]["type"], params=best_params)

        # Train
        model.fit(X_train, y_train)

        # Evaluate
        metrics, y_pred, y_proba = evaluate_model(model, X_test, y_test)
        plot_confusion_matrix(y_test, y_pred)
        plot_precision_recall_curve(y_test, y_proba)
        plot_roc_curve(y_test, y_proba)

        # Explainer
        explain_model(model, X_train)

        # Training data
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

        # Save
        save_artifacts(
            model=model,
            params=best_params,
            scaler=scaler,
            train_data=train_data_snapshot,
            metrics=metrics,
        )

        logger.info("Training pipeline completed.")

    except Exception as e:
        logger.exception("Error in training pipeline.")
        raise e
