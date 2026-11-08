import warnings

import pandas as pd
import pytest
from sktime.forecasting.model_selection import ForecastingGridSearchCV

from macroframe_forecast import MFF, MFF_mixed_freqency
from macroframe_forecast.utils import DefaultForecaster


@pytest.fixture
def data():
    return pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=pd.period_range("2000", periods=3, freq="Y"))


@pytest.mark.parametrize(
    "model_class, constraint_names",
    [
        (MFF, ("equality_constraints", "inequality_constraints")),
        (MFF_mixed_freqency, ("constraints_with_wildcard", "ineq_constraints_with_wildcard")),
    ],
)
@pytest.mark.parametrize("explicit_none", [False, True])
def test_default_constraint_lists_are_independent(data, model_class, constraint_names, explicit_none):
    model_data = {"Y": data} if model_class is MFF_mixed_freqency else data
    kwargs = dict.fromkeys(constraint_names) if explicit_none else {}
    first = model_class(model_data, **kwargs)
    second = model_class(model_data, **kwargs)
    assert getattr(first, constraint_names[0]) is not getattr(first, constraint_names[1])

    for name in constraint_names:
        first_constraints = getattr(first, name)
        second_constraints = getattr(second, name)
        assert first_constraints == second_constraints == []
        assert first_constraints is not second_constraints
        first_constraints.append("a? - 5")
        assert second_constraints == []
        assert getattr(model_class(model_data, **kwargs), name) == []


@pytest.mark.parametrize(
    "model_class, constraint_names",
    [
        (MFF, ("equality_constraints", "inequality_constraints")),
        (MFF_mixed_freqency, ("constraints_with_wildcard", "ineq_constraints_with_wildcard")),
    ],
)
@pytest.mark.parametrize("contents", [[], ["a? - 5"]])
def test_supplied_constraint_lists_are_retained(data, model_class, constraint_names, contents):
    model_data = {"Y": data} if model_class is MFF_mixed_freqency else data
    constraints = {name: contents.copy() for name in constraint_names}
    model = model_class(model_data, **constraints)
    for name, supplied in constraints.items():
        assert getattr(model, name) is supplied
        supplied.append("a? - 6")
        assert getattr(model, name) == contents + ["a? - 6"]


@pytest.mark.parametrize("explicit_none", [False, True])
def test_mixed_frequency_default_forecasters_are_independent(data, explicit_none):
    kwargs = {"forecaster": None} if explicit_none else {}
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        first = MFF_mixed_freqency({"Y": data}, **kwargs)
        second = MFF_mixed_freqency({"Y": data}, **kwargs)

    assert isinstance(first.forecaster, ForecastingGridSearchCV)
    assert isinstance(second.forecaster, ForecastingGridSearchCV)
    assert first.forecaster is not second.forecaster
    assert first.forecaster.forecaster is not second.forecaster.forecaster
    original_selection = second.forecaster.forecaster.selected_forecaster
    first.forecaster.forecaster.set_params(selected_forecaster="naive_last")
    assert second.forecaster.forecaster.selected_forecaster == original_selection
    if "remember_data" in first.forecaster.get_config():
        first.forecaster.set_config(remember_data=False)
        assert second.forecaster.get_config()["remember_data"] is True


@pytest.mark.parametrize("model_class", [MFF, MFF_mixed_freqency])
def test_supplied_forecaster_is_retained(data, model_class, monkeypatch):
    forecaster = DefaultForecaster(small_sample=True)
    original_config = forecaster.get_config()
    original_params = forecaster.get_params(deep=True)
    model_data = {"Y": data} if model_class is MFF_mixed_freqency else data

    def fail_if_called():
        pytest.fail("A supplied forecaster must not trigger default construction")

    monkeypatch.setattr("macroframe_forecast.MFF_mixed_frequency.DefaultForecaster", fail_if_called)
    model = model_class(model_data, forecaster=forecaster)
    assert model.forecaster is forecaster
    assert forecaster.get_config() == original_config
    assert forecaster.get_params(deep=True) == original_params
    assert not forecaster.is_fitted
