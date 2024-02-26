"""Contains functions for validating the data at various steps."""
from numpy import array
from pandas import DataFrame


def can_forecast(df: DataFrame) -> bool:
    """
    Check if the dataframe is suitable for forecasting. If the dataframe does
    not contain any NaN values, then there is nothing to forecast. If this
    check fails, make sure to extend time for some of the variables.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe containing time-series data.

    Returns
    -------
    bool
        Returns True if the dataframe is suitable for forecasting, False otherwise.

    Examples
    --------
    >>> df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, np.nan, 6]})
    >>> can_forecast(df)
    True

    >>> df_empty = pd.DataFrame()
    >>> can_forecast(df_empty)
    False
    """
    if len(df) == 0:
        return False
    return bool(df.isna().any().sum() > 0)


def is_consistent_shape(df: DataFrame, constraints: dict[int, array]) -> bool:
    """
    Check if the size of constraint matrices is consistent with the number of variables.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe containing variables for which constraints are applied.

    constraints : dict[int, array]
        A dictionary where keys are indices (int) and values are 2D numpy arrays
        representing constraint matrices. Each matrix should have a number of columns
        consistent with the number of variables in the dataframe.

    Returns
    -------
    bool
        Returns True if the sizes of all constraint matrices are consistent with the
        number of variables in the dataframe, False otherwise.

    Notes
    -----
    This function checks whether the size of each constraint matrix in the provided
    dictionary is consistent with the number of variables in the dataframe. If the
    dataframe is empty, it is considered inconsistent.

    Examples
    --------
    >>> df = pd.DataFrame({'var1': [1, 2, 3], 'var2': [4, 5, 6]})
    >>> C_dict = {1: np.array([[1, 0], [0, 1]]), 2: np.array([[0, 1], [1, 0]])}
    >>> is_consistent_shape(df, constraints=C_dict)
    True

    >>> df_empty = pd.DataFrame()
    >>> C_dict_empty = {}
    >>> is_consistent_shape(df_empty, constraints=C_dict_empty)
    False
    """

    if len(df) == 0:
        return False
    if len(constraints) == 0:
        # no constraints - consistent
        return True
    if len(df) != len(constraints):
        # constraints are not defined for every time period
        return False
    for constraint in constraints.values():
        n_constraint_rows, n_constraint_columns = constraint.shape
        if n_constraint_columns != len(df.columns):
            return False
        if n_constraint_rows > len(df.columns):
            # more constraints than variables in the data
            return False
    return True


def is_consistent_intercept(constraints: dict[int, array], intercepts: dict[int, array]) -> bool:
    """
    Check if the size of constraint matrices is consistent with the size of the  of variables.

    Parameters
    ----------
    constraints : dict[int, array]
        A dictionary where keys are indices (int) and values are 2D numpy arrays
        representing constraint matrices. Each matrix should have a number of columns
        consistent with the number of variables in the dataframe.

    intercepts : dict[int, array]
        A dictionary where keys are indices (int) and values are 2D numpy arrays
        representing the intercepts in matrix equation: constraints @ df = intercepts.
        Each matrix should have a number of columns consistent with the number
        of constraints.

    Returns
    -------
    bool
        Returns True if the interecept shapes are consistent with the constraint
        matrices, False otherwise.
    """

    if len(constraints) != len(intercepts):
        return False

    for constraint, intercept in zip(constraints.values(), intercepts.values()):
        n_constraint_rows, n_constraint_columns = constraint.shape
        n_intercepts, check = intercept.shape

        if check != 1:
            raise ValueError(
                "Shape of the d_dict is not consistent. Please check \
                             the values in `d_dict` have shape that is consistent\
                             with values in `C_dict`."
            )

        if n_intercepts != n_constraint_rows:
            return False

    return True
