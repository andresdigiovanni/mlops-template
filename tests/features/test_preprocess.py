import pandas as pd

from src.features import preprocess_data


def test_preprocessing_shapes():
    # Arrange
    X = pd.DataFrame([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = pd.Series([0, 1, 0, 1])

    # Act
    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        X, y, test_size=0.25, random_state=0
    )

    assert X_train.shape[0] == 3
    assert X_test.shape[0] == 1
    assert X_train.shape[1] == X_test.shape[1]
