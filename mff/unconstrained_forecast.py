from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon

from .get_default_forecaster import get_default_forecaster

import cProfile
import pstats
import io
from functools import wraps


def profile(func):
    """A decorator that uses cProfile to profile a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()

        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

        return result

    return wrapper


# @profile
def unconstrained_forecast(
        df: pd.DataFrame, Tin: int, forecaster: Optional[BaseForecaster] = None, fh: ForecastingHorizon or int = None
) -> tuple[pd.DataFrame, BaseForecaster, ForecastingHorizon]:
    if forecaster is None:
        forecaster = get_default_forecaster(Tin=Tin)

    n_na = df.isna().sum(axis=1)
    if not ((n_na == 0) | (n_na == df.shape[0])).any():
        yf = df
        Xf = None
        Xp = None
    else:
        unknown_variables = df.columns[df.isna().sum(axis=0) > 0]
        known_variables = df.columns[df.isna().sum(axis=0) == 0]
        mask_fit = df[unknown_variables].notna().sum(axis=1) > 0
        mask_predict = df[unknown_variables].isna().sum(axis=1) > 0

        Xf = df.loc[mask_fit, known_variables]
        yf = df.loc[mask_fit, unknown_variables]

        Xp = df.loc[mask_predict, known_variables].reset_index(drop=True)

    # yp = df.loc[mask_predict, unknown_variables]
    if isinstance(fh, (int, np.int64)):
        fh = ForecastingHorizon(values=range(1, fh + 1), is_relative=True)
    elif isinstance(fh, ForecastingHorizon):
        fh = fh
    else:
        fh = ForecastingHorizon(values=Xp.index + 1, is_relative=True)

    yp = forecaster.fit_predict(y=yf, X=Xf, fh=fh, X_pred=Xp)

    df1 = df.copy()
    if Xp is None:
        df1 = pd.concat([df1, yp])
    else:
        df1.loc[mask_predict, unknown_variables] = yp

    return df1, forecaster, fh
