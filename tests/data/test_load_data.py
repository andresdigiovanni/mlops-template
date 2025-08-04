import pandas as pd

from src.data import load_data


def test_load_data_shape():
    # Act
    X, y = load_data()

    # Assert
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
