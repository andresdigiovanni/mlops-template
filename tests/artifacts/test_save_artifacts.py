import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.artifacts import save_artifacts


def test_save_artifacts(tmp_path):
    # Arrange
    model = LogisticRegression()
    scaler = StandardScaler()
    train_data = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    metrics = {"accuracy": 0.95}
    config = {"model": {"type": "lr"}}
    confusion_matrix = np.array([[10, 1], [0, 9]])

    # Act
    tmp_path_artifacts = save_artifacts(
        model, scaler, train_data, metrics, config, confusion_matrix, tmp_path
    )

    # Assert
    assert (tmp_path_artifacts / "model.pkl").exists()
    assert (tmp_path_artifacts / "scaler.pkl").exists()
    assert (tmp_path_artifacts / "metrics.json").exists()
    assert (tmp_path_artifacts / "config_used.yaml").exists()
    assert (tmp_path_artifacts / "confusion_matrix.png").exists()
