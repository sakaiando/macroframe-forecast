import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon
from .unconstrained_forecast import delete_exogenous_islands, staggered_forecast
from .step2 import Reconciler


def calculate_residuals(df, forecaster, fcast_horizons, cols=None):
    if cols is None:
        cols = df.cols
    resids = []
    for i in df.index[:-fcast_horizons]:
        forecast_horizon = ForecastingHorizon(range(i + 1, i + fcast_horizons + 1), is_relative=False)
        y_hat_multi = forecaster.predict(forecast_horizon)
        resid = (y_hat_multi - df).dropna().reset_index(drop=True).stack()
        resids.append(resid)
    resids = pd.concat(resids, axis=1).T

    W_idx = Reconciler.multiindex_from_multiindex_product(
        pd.Index(range(df.index.max() + 1, df.index.max() + fcast_horizons + 1), name='year'), cols)
    resids.columns = W_idx
    return resids


def calculate_oos_residuals(df, forecaster, fh, col_dict=None, n_periods=2):
    df = delete_exogenous_islands(df)
    resids = []
    for h in range(fh, fh + n_periods):
        df_est = df[~df.shift(-h).isna()].dropna(how='all')
        df_pred, _, _ = staggered_forecast(df_est, 10, fh=fh)
        resid = (df - df_pred)[df_est.isna()].shift(h).stack().dropna()
        resids.append(resid)
    resids = pd.concat(resids, axis=1).T

    temp_idx = resids.columns.get_level_values(1).map(col_dict)
    temp_idx.names = ['variable', 'freq', 'subperiod']
    idx = concatenate_index_on_levels(resids.columns, temp_idx).droplevel(level=1)
    resids.columns = idx

    return resids


def raw_covariance(resids):
    return resids.cov()


def concatenate_index_on_levels(index1, index2):
    """
    Concatenates two MultiIndexes on the levels dimension.

    Parameters:
    index1 (pd.MultiIndex): The first MultiIndex.
    index2 (pd.MultiIndex): The second MultiIndex.

    Returns:
    pd.MultiIndex: The concatenated MultiIndex.
    """
    df1 = index1.to_frame(index=False)
    df2 = index2.to_frame(index=False)

    # Concatenate the DataFrames
    concatenated_df = pd.concat([df1, df2], axis=1)

    # Convert the concatenated DataFrame back to a MultiIndex
    new_index = pd.MultiIndex.from_frame(concatenated_df)
    return new_index


if __name__ == '__main__':
    pass
    # Put unit tests here
    # resids = calculate_residuals(data_nona, forecaster, n_horizons)
    # W = raw_covariance(resids)