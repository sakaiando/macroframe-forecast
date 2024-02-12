from numpy import array
from mff.validators import can_forecast, is_consistent_shape
from pandas import DataFrame

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
    no_constraints = dict({"a": array([0,0,0])})
    assert is_consistent_shape(empty_dataframe, no_constraints) is False


    # no constraints
    mixed_data = DataFrame({"a": [0, 1, 2, 3], "b": [1, None, 2, 3]})
    no_constraints = dict()
    assert is_consistent_shape(mixed_data, no_constraints) is True

    