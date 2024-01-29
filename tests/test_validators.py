from mff.validators import can_forecast, is_consistent_shape
from pandas import DataFrame

def test_can_forecast():

    empty_dataframe = DataFrame()
    assert can_forecast(empty_dataframe) is False

    mixed_data = DataFrame({"a": [0, 1, 2, 3], "b": [1, None, 2, 3]})

    assert can_forecast(mixed_data) is True



def test_is_consistent_shape():

    empty_dataframe = DataFrame()
    no_constraints = dict()
    assert is_consistent_shape(empty_dataframe, no_constraints) is False

    # does this scenario make sense? How should we handle the case of no constraints
    mixed_data = DataFrame({"a": [0, 1, 2, 3], "b": [1, None, 2, 3]})
    no_constraints = dict()
    assert is_consistent_shape(mixed_data, no_constraints) is False

    