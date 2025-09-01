import logging
from pathlib import Path

from src.artifacts import save_artifact
from src.data import load_data, normalize_column_names
from src.evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from src.experiment_tracker import create_experiment_tracker
from src.explainer import explain_model
from src.features import prepare_training_data, preprocess_data
from src.models import create_model
from src.target import encode_target
from src.tuning import create_tuner


def run_training_pipeline(cfg) -> None:
    logger = logging.getLogger()
    logger.info("Starting ML training pipeline")

    artifacts_path = Path(cfg["artifacts"]["path"])

    try:
        # Load data
        X, y = load_data(cfg["data"]["path"], cfg["data"]["target"])
        X = normalize_column_names(X)
        y = encode_target(y)

        # Base model
        model = create_model(model_type=cfg["model"]["type"])

        # Prepare training data
        X = preprocess_data(X)
        X_train, X_test, y_train, y_test, transformer, transformations = (
            prepare_training_data(
                model,
                X,
                y,
                scoring=cfg["scoring"],
                test_size=cfg["data"]["test_size"],
                random_state=cfg["data"]["random_state"],
            )
        )

        # Hyperparameter tuner
        tuner = create_tuner(
            cfg["model"]["type"], X_train, y_train, scoring=cfg["scoring"]
        )
        best_params = tuner.run()

        # Model creation
        model = create_model(model_type=cfg["model"]["type"], params=best_params)

        # Train
        model.fit(X_train, y_train)

        # Evaluate
        metrics, y_test_pred, y_test_proba = evaluate_model(model, X_test, y_test)
        plot_confusion_matrix(y_test, y_test_pred, base_path=artifacts_path)
        plot_precision_recall_curve(y_test, y_test_proba, base_path=artifacts_path)
        plot_roc_curve(y_test, y_test_proba, base_path=artifacts_path)

        # Explainer
        explain_model(
            model, X_train, cfg["model"]["explainer"], base_path=artifacts_path
        )

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

        # Save artifacts to local directory
        save_artifact(model, artifacts_path / "model.pkl")
        save_artifact(best_params, artifacts_path / "params.json")
        save_artifact(transformer, artifacts_path / "transformer.pkl")
        save_artifact(transformations, artifacts_path / "transformations.json")
        save_artifact(train_data_snapshot, artifacts_path / "train_data.csv")
        save_artifact(metrics, artifacts_path / "metrics.json")

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
                artifacts_path / "transformer.pkl", category="artifact"
            )
            tracker.log_artifact(
                artifacts_path / "transformations.json", category="artifact"
            )
            tracker.log_artifact(artifacts_path / "train_data.csv", category="dataset")
            tracker.log_artifact(artifacts_path / "cm.png", category="images")
            tracker.log_artifact(artifacts_path / "pr_curve.png", category="images")
            tracker.log_artifact(artifacts_path / "roc_curve.png", category="images")
            tracker.log_artifact(artifacts_path / "shap_summary.png", category="images")

        logger.info("Training pipeline completed.")

    except Exception as e:
        logger.exception("Error in training pipeline.")
        raise e
