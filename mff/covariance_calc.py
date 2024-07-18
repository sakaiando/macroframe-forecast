import pandas as pd
from sktime.forecasting.base import ForecastingHorizon

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


def raw_covariance(resids):
    return resids.cov()


if __name__ == '__main__':
    pass
    # Put unit tests here
    # resids = calculate_residuals(data_nona, forecaster, n_horizons)
    # W = raw_covariance(resids)