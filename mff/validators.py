"""Contains functions for validating the data at various steps."""
from numpy import array
from pandas import DataFrame


def can_forecast(df: DataFrame) -> bool:
    """If the dataframe does not contain any nan values, then there is nothing
    to forecast. If this check fails, make sure to extend time for some of the
    variables."""
    if len(df) == 0:
        return False
    return bool(df.isna().any().sum() > 0)


def is_consistent_shape(df: DataFrame, C_dict: dict[int, array]) -> bool:
    """check 2: make sure the size of contraint matrix is consistent with number of vars"""
    for i in C_dict:
        if C_dict[i].shape[1] != len(df.columns):
            return False
    return True
