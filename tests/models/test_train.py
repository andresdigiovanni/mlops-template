import numpy as np
from sklearn.linear_model import LogisticRegression

from src.models import train_model


def test_train_model_fit():
    # Arrange
    X = np.array([[0, 1], [1, 1], [1, 0]])
    y = np.array([0, 1, 0])
    model = LogisticRegression()

    # Act
    trained_model = train_model(model, X, y)
    preds = trained_model.predict(X)

    # Assert
    assert hasattr(trained_model, "predict")
    assert len(preds) == len(y)
