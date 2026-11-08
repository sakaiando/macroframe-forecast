from numpy.random import sample
from pandas import DataFrame, Index, PeriodIndex, Series
from pytest import mark
from sktime.forecasting.trend import PolynomialTrendForecaster

from macroframe_forecast.utils import (
    FillAllEmptyCells,
    FillAnEmptyCell,
    GenPredTrueData,
    GenWeightMatrix,
    expand_wildcard,
    find_permissible_wildcard,
    find_strings_to_replace_wildcard,
    get_freq_of_freq,
)


@mark.parametrize("method", ["identity", "oas", "oasd", "monotone diagonal"])
def test_weight_matrix_methods_preserve_cell_order_and_covariance(method):
    import numpy as np
    import pandas as pd
    from sklearn.covariance import OAS

    columns = pd.MultiIndex.from_tuples([("a", 2020), ("a", 2021), ("b", 2020)])
    errors = np.array([[2, 0.2, 3], [4, 0.1, 4], [8, 0.3, 2], [6, 0.5, 1], [10, 0.4, 5]], dtype=float)
    truth = pd.DataFrame(10.0, index=range(len(errors)), columns=columns)
    pred = truth + errors
    pred_list = [pred.iloc[:, :2], pred.iloc[:, 2:]]
    true_list = [truth.iloc[:, :2], truth.iloc[:, 2:]]
    covariance = pd.DataFrame(errors, columns=columns).cov().to_numpy()

    weight, shrinkage = GenWeightMatrix(pred_list, true_list, shrinkage_method=method)

    assert weight.index.equals(columns)
    assert weight.columns.equals(columns)
    if method == "identity":
        expected = np.eye(3)
        assert np.isnan(shrinkage)
    elif method == "oas":
        estimator = OAS().fit(errors)
        expected = estimator.covariance_
        assert np.isclose(shrinkage, estimator.shrinkage_)
    elif method == "oasd":
        diagonal = np.diag(np.diag(covariance))
        numerator = np.trace(covariance @ covariance) - np.trace(diagonal @ diagonal)
        denominator = numerator + np.trace(covariance) ** 2 - np.trace(diagonal @ diagonal)
        expected_shrinkage = min(denominator / (len(errors) * numerator), 1)
        expected = (1 - expected_shrinkage) * covariance + expected_shrinkage * diagonal
        assert np.isclose(shrinkage, expected_shrinkage)
    else:
        variances = np.diag(covariance).copy()
        variances[:2] = np.maximum.accumulate(variances[:2])
        expected = np.diag(variances)
        assert np.isnan(shrinkage)
    np.testing.assert_allclose(weight, expected)
    np.testing.assert_allclose(weight, weight.T)
    assert np.linalg.eigvalsh(weight).min() > 0


@mark.parametrize("method", ["oasd", "monotone diagonal"])
def test_single_cell_weight_uses_sample_variance(method):
    import numpy as np
    import pandas as pd

    columns = pd.MultiIndex.from_tuples([("a", 2020)])
    pred = pd.DataFrame([1.0, 2.0, 4.0], columns=columns)
    true = pd.DataFrame(0.0, index=pred.index, columns=columns)

    weight, shrinkage = GenWeightMatrix([pred], [true], shrinkage_method=method)

    pd.testing.assert_frame_equal(weight, pred.cov())
    assert np.isnan(shrinkage)


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


def test_fill_an_empty_cell_uses_target_horizon():
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(
        {"a": 10.0 * np.arange(1, 41), "b": 1.0 * np.arange(1, 41)},
        index=pd.date_range("1986", periods=40, freq="YE").year,
    )
    df.iloc[-3:, 0] = np.nan
    forecaster = PolynomialTrendForecaster(degree=1)

    for row, expected in zip(df.index[-3:], [380.0, 390.0, 400.0]):
        prediction, _ = FillAnEmptyCell(df, row, "a", forecaster)
        assert len(prediction) == 1
        assert np.isclose(prediction.iloc[0], expected)


def test_first_stage_forecasts_are_not_flattened():
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(
        {"a": 10.0 * np.arange(1, 41), "b": 1.0 * np.arange(1, 41)},
        index=pd.date_range("1986", periods=40, freq="YE").year,
    )
    df.iloc[-3:, 0] = np.nan
    forecaster = PolynomialTrendForecaster(degree=1)

    filled, _ = FillAllEmptyCells(df, forecaster, parallelize=False)
    pred, true, model = GenPredTrueData(df, forecaster, n_forecast_error=2, parallelize=False)

    assert np.allclose(filled.iloc[-3:, 0], [380.0, 390.0, 400.0])
    assert np.allclose(pred, true)
    assert pred.shape == true.shape == model.shape == (2, 3)
    assert list(pred.columns) == [("a", year) for year in df.index[-3:]]
