from numpy import array, ones
from pandas import DataFrame
from pytest import raises

from mff.validators import can_forecast, is_consistent_intercept, is_consistent_shape


def test_can_forecast():
    empty_dataframe = DataFrame()
    assert can_forecast(empty_dataframe) is False

    mixed_data = DataFrame({"a": [0, 1, 2, 3], "b": [1, None, 2, 3]})

    assert can_forecast(mixed_data) is True


def test_is_consistent_shape():
    # return false if dataframe is empty
    empty_dataframe = DataFrame()
    no_constraints = dict()
    assert is_consistent_shape(empty_dataframe, no_constraints) is False

    # return false if dataframe is empty, even with constraints
    empty_dataframe = DataFrame()
    no_constraints = dict({"a": array([0, 0, 0])})
    assert is_consistent_shape(empty_dataframe, no_constraints) is False

    # no constraints
    mixed_data = DataFrame({"a": [0, 1, 2, 3], "b": [1, None, 2, 3]})
    no_constraints = dict()
    assert is_consistent_shape(mixed_data, no_constraints) is True

    # consistent constraints
    mixed_data = DataFrame({"a": [0, 1, 2, 3], "b": [1, 2, 3, None]})
    constraints = {t: ones(shape=(2, 2)) for t in range(len(mixed_data))}
    assert is_consistent_shape(mixed_data, constraints) is True

    # inconsistent constraints - insufficient variables
    mixed_data = DataFrame({"a": [0, 1, 2, 3], "b": [1, 2, 3, None]})
    constraints = {t: ones(shape=(2, 1)) for t in range(len(mixed_data))}
    assert is_consistent_shape(mixed_data, constraints) is False

    # inconsistent constraints - insufficient time periods
    mixed_data = DataFrame({"a": [0, 1, 2, 3], "b": [1, 2, 3, None]})
    constraints = {t: ones(shape=(2, 1)) for t in range(len(mixed_data))}
    assert is_consistent_shape(mixed_data, constraints) is False

    # inconsistent constraints - more constraints than variables
    mixed_data = DataFrame({"a": [0, 1, 2, 3], "b": [1, 2, 3, None]})
    constraints = {t: ones(shape=(4, 2)) for t in range(len(mixed_data))}
    assert is_consistent_shape(mixed_data, constraints) is False


def test_is_consistent_intercept():
    # empty constraints
    no_constraints = dict()
    no_intercepts = dict()
    assert is_consistent_intercept(no_constraints, no_intercepts) is True

    # mismatched constraint sizes
    constraints = {t: ones(shape=(2, 5)) for t in range(4)}
    no_intercepts = dict()
    assert is_consistent_intercept(constraints, no_intercepts) is False

    # matching constraint sizes
    constraints = {t: ones(shape=(2, 5)) for t in range(4)}
    intercepts = {t: ones(shape=(2, 1)) for t in range(4)}
    assert is_consistent_intercept(constraints, intercepts) is True

    # mismatching intercept sizes
    constraints = {t: ones(shape=(2, 5)) for t in range(4)}
    intercepts = {t: ones(shape=(2, 3)) for t in range(4)}
    with raises(ValueError):
        is_consistent_intercept(constraints, intercepts)
