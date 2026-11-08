import subprocess
import sys
import warnings
from typing import cast

import numpy as np
import pandas as pd
import pytest
import sktime
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster

from macroframe_forecast import MFF
from macroframe_forecast.utils import DefaultForecaster, FillAllEmptyCells, FillAnEmptyCell, GenPredTrueData


REMEMBER_DATA_WARNING = r"The default of config ``remember_data`` will change"


@pytest.fixture
def forecast_data():
    df = pd.DataFrame({"a": np.arange(1.0, 21.0) ** 2}, index=pd.period_range("2000", periods=20, freq="Y"))
    df.iloc[-2:] = np.nan
    return df


def test_import_does_not_emit_remember_data_warning():
    result = subprocess.run(
        [
            sys.executable,
            "-W",
            f"error:{REMEMBER_DATA_WARNING}:FutureWarning",
            "-c",
            "import macroframe_forecast",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("small_sample", [True, False])
def test_default_forecaster_retains_data_without_warning(forecast_data, small_sample):
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        forecaster = DefaultForecaster(small_sample=small_sample)
    if not small_sample:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", REMEMBER_DATA_WARNING, FutureWarning)
            forecaster.set_params(param_grid={"selected_forecaster": ["naive_last"]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error", FutureWarning)
        prediction, fitted = FillAnEmptyCell(forecast_data, forecast_data.index[-1], "a", forecaster)

    assert not any(isinstance(warning.message, FutureWarning) for warning in caught)
    assert prediction.iloc[0] == forecast_data["a"].iloc[-3]
    if "remember_data" in fitted.get_config():
        assert fitted.get_config()["remember_data"] is True
    pd.testing.assert_frame_equal(pd.DataFrame(fitted._y), forecast_data.iloc[:-2])


def test_default_forecaster_retains_data_after_sktime_default_changes(monkeypatch, forecast_data):
    if "remember_data" not in DefaultForecaster(small_sample=True).get_config():
        pytest.skip("sktime does not support remember_data")
    monkeypatch.setattr(BaseForecaster, "_config", {**BaseForecaster._config, "remember_data": False})
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        forecaster = DefaultForecaster(small_sample=True)
        _, fitted = FillAnEmptyCell(forecast_data, forecast_data.index[-1], "a", forecaster)
    assert forecaster.get_config()["remember_data"] is True
    pd.testing.assert_frame_equal(pd.DataFrame(fitted._y), forecast_data.iloc[:-2])


@pytest.mark.parametrize("remember_data", [True, False])
@pytest.mark.parametrize("clone_config", [True, False])
def test_supplied_forecaster_config_and_predictions_are_preserved(forecast_data, remember_data, clone_config):
    if not remember_data and sktime.__version__ == "0.28.0":
        pytest.skip("sktime 0.28.0 loses the cutoff with remember_data=False, even without macroframe-forecast")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        forecaster = PolynomialTrendForecaster(degree=1)
        if "remember_data" not in forecaster.get_config() and not remember_data:
            pytest.skip("sktime does not support remember_data=False")
        if "remember_data" in forecaster.get_config():
            forecaster.set_config(remember_data=remember_data)
        forecaster.set_config(clone_config=clone_config)
        reference = PolynomialTrendForecaster(degree=1).fit(forecast_data["a"].iloc[:-2], fh=[2])
    original_config = forecaster.get_config()

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        prediction, fitted = FillAnEmptyCell(forecast_data, forecast_data.index[-1], "a", forecaster)
        filled, models = FillAllEmptyCells(forecast_data, forecaster, parallelize=False)
        pred, true, historical_models = GenPredTrueData(
            forecast_data, forecaster, n_forecast_error=2, parallelize=False
        )

    pd.testing.assert_series_equal(prediction, cast(pd.Series, reference.predict()))
    assert forecaster.get_config() == original_config
    assert not forecaster.is_fitted
    assert fitted is not forecaster
    assert filled.notna().all().all()
    assert pred.shape == true.shape == historical_models.shape == (2, 2)
    for model in [fitted, *models.iloc[-2:, 0], *historical_models.to_numpy().ravel()]:
        if "remember_data" in original_config:
            assert model.get_config()["remember_data"] is remember_data
    if remember_data:
        pd.testing.assert_frame_equal(pd.DataFrame(fitted._y), forecast_data.iloc[:-2])
        new_data = pd.Series([500.0], index=forecast_data.index[-2:-1], name="a")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            warnings.filterwarnings("ignore", REMEMBER_DATA_WARNING, FutureWarning)
            fitted.update(new_data)
            reference.update(new_data)
            pd.testing.assert_series_equal(cast(pd.Series, fitted.predict()), cast(pd.Series, reference.predict()))


def test_unrelated_future_warnings_are_not_silenced(forecast_data, monkeypatch):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        forecaster = PolynomialTrendForecaster()

    original_clone = forecaster.clone

    def warn_on_clone():
        warnings.simplefilter("default", FutureWarning)
        warnings.warn("An unrelated change", FutureWarning)
        return original_clone()

    monkeypatch.setattr(forecaster, "clone", warn_on_clone)
    filters_before = warnings.filters[:]
    with pytest.warns(FutureWarning, match="An unrelated change"):
        FillAnEmptyCell(forecast_data, forecast_data.index[-1], "a", forecaster)
    assert warnings.filters == filters_before


def test_warning_filters_are_restored_on_failure(forecast_data, monkeypatch):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        forecaster = PolynomialTrendForecaster()

    def fail_on_clone():
        warnings.simplefilter("default", FutureWarning)
        warnings.warn("An unrelated change", FutureWarning)
        raise ValueError("Cannot clone")

    monkeypatch.setattr(forecaster, "clone", fail_on_clone)
    filters_before = warnings.filters[:]
    with pytest.warns(FutureWarning, match="An unrelated change"):
        with pytest.raises(ValueError, match="Cannot clone"):
            FillAnEmptyCell(forecast_data, forecast_data.index[-1], "a", forecaster)
    assert warnings.filters == filters_before


@pytest.mark.parametrize("use_default", [True, False])
def test_mff_fit_does_not_emit_remember_data_warning(forecast_data, use_default):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        forecaster = None if use_default else PolynomialTrendForecaster(degree=1)
    model = MFF(
        forecast_data,
        forecaster=forecaster,
        parallelize=False,
        n_forecast_error=2,
        shrinkage_method="identity",
        default_lam=0,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error", FutureWarning)
        result = model.fit()

    assert not any(isinstance(warning.message, FutureWarning) for warning in caught)
    assert result.notna().all().all()
    pd.testing.assert_frame_equal(result.iloc[:-2], forecast_data.iloc[:-2])
    np.testing.assert_allclose(result.iloc[-2:], model.df1.iloc[-2:])
