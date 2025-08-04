import numpy as np
from sklearn.linear_model import LogisticRegression

from src.evaluation import evaluate_model


def test_evaluation_metrics():
    # Arrange
    X = np.array([[0, 1], [1, 1], [1, 0]])
    y = np.array([0, 1, 0])
    model = LogisticRegression().fit(X, y)

    # Act
    metrics, cm = evaluate_model(model, X, y)

    # Assert
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert cm.shape == (2, 2)
