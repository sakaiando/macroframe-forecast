from numpy.random import sample
from pandas import DataFrame, Index, PeriodIndex, Series

from macroframe_forecast.utils import (
    expand_wildcard,
    find_permissible_wildcard,
    find_strings_to_replace_wildcard,
    get_freq_of_freq,
)


def test_find_permissible_wildcard():
    assert find_permissible_wildcard(["a", "b", "c"], _seed=0) == "m"
    assert find_permissible_wildcard(["a", "b", "c"], _seed=10) == "s"


def test_find_strings_to_replace_wildcard():
    n = 30
    p = 2
    years = [str(y) for y in range(2000, 2000 + n)]
    df = DataFrame(sample([n, p]), columns=["a", "b"], index=years)
    df0_stacked = df.T.stack()
    all_cells_index = df0_stacked.index
    var_list = Series([f"{a}_{b}" for a, b in all_cells_index], index=all_cells_index)
    constraint = "ax + bx"
    wildcard = "x"
    missing_string_list = find_strings_to_replace_wildcard(constraint, var_list, wildcard)
    assert missing_string_list == [f"_{y}" for y in years]


def test_expand_wildcard():
    import numpy as np
    import pandas as pd

    n = 30
    p = 2
    years = [str(y) for y in range(2000, 2000 + n)]
    df = pd.DataFrame(np.random.sample([n, p]), columns=["a", "b"], index=years)
    df0_stacked = df.T.stack()
    all_cells_index = df0_stacked.index
    var_list = pd.Series([f"{a}_{b}" for a, b in all_cells_index], index=all_cells_index)
    constraints_with_alphabet_wildcard = ["ax + bx"]
    alphabet_wildcard = "x"
    constraints = expand_wildcard(constraints_with_alphabet_wildcard, var_list=var_list, wildcard=alphabet_wildcard)
    assert constraints == [f"a_{y} + b_{y}" for y in years]


def test_get_freq_of_freq_quarterly():
    years = [2000, 2000, 2001]
    quarters = [1, 2, 4]

    test_index = PeriodIndex.from_fields(year=years, quarter=quarters)

    assert get_freq_of_freq(test_index, "Y").equals(Index(years, dtype="int64"))
    assert get_freq_of_freq(test_index, "Q").equals(Index(quarters, dtype="int64"))


def test_get_freq_of_freq_datetime():
    years = [2000, 2000, 2001]
    months = [1, 6, 11]
    days = [3, 10, 10]
    hours = [4, 6, 10]
    minutes = [5, 10, 30]
    seconds = [1, 4, 5]

    test_index_2 = PeriodIndex.from_fields(
        year=years, month=months, day=days, hour=hours, minute=minutes, second=seconds, freq="s"
    )
    assert get_freq_of_freq(test_index_2, "M").equals(Index(months, dtype="int64"))
    assert get_freq_of_freq(test_index_2, "W").equals(Index([1, 23, 45], dtype="int32"))
    assert get_freq_of_freq(test_index_2, "D").equals(Index(days, dtype="int32"))
    assert get_freq_of_freq(test_index_2, "H").equals(Index(hours, dtype="int32"))
    assert get_freq_of_freq(test_index_2, "T").equals(Index(minutes, dtype="int32"))
    assert get_freq_of_freq(test_index_2, "S").equals(Index(seconds, dtype="int32"))
