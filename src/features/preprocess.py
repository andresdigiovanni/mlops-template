import logging

import pandas as pd

logger = logging.getLogger()


def preprocess_data(X: pd.DataFrame) -> pd.DataFrame:
    map_months = {
        "jan": 0,
        "feb": 1,
        "mar": 2,
        "apr": 3,
        "may": 4,
        "jun": 5,
        "jul": 6,
        "ago": 7,
        "sep": 8,
        "oct": 9,
        "nov": 10,
        "dec": 11,
    }
    X["month"] = X["month"].map(map_months)

    return X
