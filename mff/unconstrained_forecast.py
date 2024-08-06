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


def find_forecast_start_by_col(df):
    return df.isna().sort_index().idxmax(axis=0).replace({df.index.min(): df.index.max() + 1})


def delete_exogenous_islands(df):
    df = df.copy()
    last_data = find_forecast_start_by_col(df)
    for col in last_data.index:
        df.loc[last_data[col]:, col] = np.nan
    return df


def staggered_forecast(df, Tin=10, fh=None, forecaster=None, add_extra_year=False):
    df = delete_exogenous_islands(df)

    step_dates = find_forecast_start_by_col(df).unique() - 1  # TODO: do we want -1 here or a more clever way?
    step_dates.sort()
    step_dates = step_dates[1:]

    # add additional year for smoothing
    if add_extra_year:
        extend_date = df.index.max() + 1
        df.loc[extend_date] = np.nan
        step_dates = np.append(step_dates, df.index.max())

    fcast_dict = {}
    fh_dict = {}
    y_hat = df.copy()
    for step in step_dates:
        y_hat_temp, fcast_dict[step], fh_dict[step] = unconstrained_forecast(df.loc[:step],
                                                                                   Tin=Tin,
                                                                                   fh=fh,
                                                                                   forecaster=forecaster
                                                                                   )
        y_hat.update(y_hat_temp)
    return y_hat, fcast_dict, fh_dict


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
        known_variables = df.columns.drop(unknown_variables)  # df.columns[df.isna().sum(axis=0) == 0]
        mask_fit = df[unknown_variables].notna().sum(axis=1) == len(unknown_variables)
        mask_predict = ~mask_fit

        Xf = df.loc[mask_fit, known_variables]
        yf = df.loc[mask_fit, unknown_variables]

        Xp = df.loc[mask_predict, known_variables].reset_index(drop=True)

    # yp = df.loc[mask_predict, unknown_variables]
    if Xp is not None:
        fh = ForecastingHorizon(values=Xp.index + 1, is_relative=True)
    elif isinstance(fh, (int, np.int64)):
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
        df1.update(yp)  # doesn't replace filled values

    return df1, forecaster, fh


if __name__ == '__main__':
    import pandas as pd
    import os

    os.chdir(r'C:\Users\dbilgin\OneDrive - International Monetary Fund (PRD)\prototype')
    df = pd.read_excel(r'.\data\input.xlsx', sheet_name='data', index_col=0).T
    Tin = 10
    staggered_forecast(df, Tin)
