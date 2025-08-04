from sklearn.linear_model import LogisticRegression

from src.tuning import BaseModelTuner


class LogisticRegressionTuner(BaseModelTuner):
    def _build_model(self, trial):
        params = {
            "C": trial.suggest_float("C", 1e-4, 10, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
            "max_iter": 1000,
        }
        params.update(
            {
                "solver": (
                    "liblinear"
                    if params["penalty"] == "l1"
                    else trial.suggest_categorical(
                        "solver", ["liblinear", "lbfgs", "saga"]
                    )
                ),
            }
        )

        return LogisticRegression(**params)
