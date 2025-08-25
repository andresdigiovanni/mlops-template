import logging

from src.artifacts import save_artifacts
from src.data import load_data, normalize_column_names
from src.evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from src.experiment_tracker import create_experiment_tracker
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
        metrics, y_test_pred, y_test_proba = evaluate_model(model, X_test, y_test)
        plot_confusion_matrix(y_test, y_test_pred)
        plot_precision_recall_curve(y_test, y_test_proba)
        plot_roc_curve(y_test, y_test_proba)

        # Explainer
        explain_model(model, X_train, cfg["model"]["explainer"])

        # Training data
        y_train_pred = model.predict(X_train)
        y_train_proba = (
            model.predict_proba(X_train)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        train_data_snapshot = X_train.copy()
        train_data_snapshot["target"] = y_train
        train_data_snapshot["pred"] = y_train_pred
        train_data_snapshot["proba"] = y_train_proba

        # Save
        base_path = save_artifacts(
            model=model,
            params=best_params,
            scaler=scaler,
            train_data=train_data_snapshot,
            metrics=metrics,
        )

        # Experiment tracking
        tracker = create_experiment_tracker(
            cfg["experiment_tracker"]["type"], cfg["experiment_tracker"]["params"]
        )

        with tracker.start_run():
            tracker.log_params(best_params)
            tracker.log_metrics(metrics)

            tracker.save_model(
                model,
                name="model",
                registered_model_name=cfg["model"]["name"],
                input_example=X_train,
                output_example=y_train_pred,
            )

            tracker.log_artifact(
                str(base_path / "scaler.pkl"), artifact_path="artifact"
            )
            tracker.log_artifact(
                str(base_path / "train_data.csv"), artifact_path="dataset"
            )
            tracker.log_artifact(str(base_path / "cm.png"), artifact_path="images")
            tracker.log_artifact(
                str(base_path / "pr_curve.png"), artifact_path="images"
            )
            tracker.log_artifact(
                str(base_path / "roc_curve.png"), artifact_path="images"
            )
            tracker.log_artifact(
                str(base_path / "shap_summary.png"), artifact_path="images"
            )

        logger.info("Training pipeline completed.")

    except Exception as e:
        logger.exception("Error in training pipeline.")
        raise e
