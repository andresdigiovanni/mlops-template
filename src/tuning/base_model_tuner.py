from abc import ABC, abstractmethod

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score


class BaseModelTuner(ABC):
    def __init__(
        self,
        X,
        y,
        n_trials=50,
        scoring="roc_auc",
        cv_splits=5,
        direction="maximize",
        random_state=42,
    ):
        self.X = X
        self.y = y
        self.n_trials = n_trials
        self.scoring = scoring
        self.direction = direction
        self.cv = StratifiedKFold(
            n_splits=cv_splits, shuffle=True, random_state=random_state
        )
        self.study = None

    def run(self):
        self.study = optuna.create_study(direction=self.direction)
        self.study.optimize(self._objective, n_trials=self.n_trials)
        return self._best_params()

    def get_best_score(self) -> float:
        return self.study.best_value

    def get_best_params(self) -> dict:
        return self._best_params()

    def _objective(self, trial):
        model = self._build_model(trial)
        score = cross_val_score(
            model, self.X, self.y, scoring=self.scoring, cv=self.cv, n_jobs=-1
        )
        return np.mean(score)

    @abstractmethod
    def _build_model(self, trial):
        pass

    @abstractmethod
    def _best_params(self):
        pass
