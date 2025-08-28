from sklearn.linear_model import LogisticRegression

from src.tuning import BaseModelTuner


class LogisticRegressionTuner(BaseModelTuner):
    def _build_model(self, trial):
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])

        if penalty == "l1":
            solver = trial.suggest_categorical("solver_l1", ["liblinear", "saga"])
        else:  # penalty == "l2"
            solver = trial.suggest_categorical(
                "solver_l2", ["liblinear", "lbfgs", "saga"]
            )

        params = {
            "C": trial.suggest_float("C", 1e-4, 10, log=True),
            "penalty": penalty,
            "solver": solver,
            "max_iter": 1000,
        }

        return LogisticRegression(**params)

    def _best_params(self):
        def _map_param(param):
            if param in ["solver_l1", "solver_l2"]:
                return "solver"
            return param

        return {_map_param(k): v for k, v in self.study.best_params.items()}
