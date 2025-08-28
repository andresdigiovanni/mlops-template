import pandas as pd


def encode_target(y: pd.Series) -> pd.Series:
    """
    Encode target variable to binary format.

    Args:
        y (pd.Series): Target variable with string labels.

    Returns:
        pd.Series: Encoded target variable with 0 and 1.
    """
    return y.apply(lambda x: 1 if x == "yes" else 0)
