import lightgbm as lgb
import pytest
from sklearn.linear_model import LogisticRegression

from src.models import create_model


def test_get_logistic_model():
    # Act
    model = create_model("lr", {"max_iter": 200})

    # Assert
    assert isinstance(model, LogisticRegression)
    assert model.max_iter == 200


def test_get_lightgbm_default():
    # Act
    model = create_model("lightgbm")

    # Assert
    assert isinstance(model, lgb.LGBMClassifier)


def test_model_not_found():
    with pytest.raises(RuntimeError):
        create_model("unsupported_model_type")
