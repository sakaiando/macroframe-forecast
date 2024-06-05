from typing import Optional

from pandas import DataFrame
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon

from mff.get_default_estimators import get_default_forecaster


def unconstrained_forecast(
    df: DataFrame, Tin: int, forecaster: Optional[BaseForecaster] = None
) -> tuple[DataFrame, BaseForecaster]:
    if forecaster is None:
        forecaster = get_default_forecaster(Tin=Tin)

    unknown_variables = df.columns[df.isna().sum(axis=0) > 0]
    known_variables = df.columns[df.isna().sum(axis=0) == 0]
    mask_fit = df[unknown_variables].notna().index
    mask_predict = df[unknown_variables].isna().index

    Xf = df.loc[mask_fit, known_variables]
    yf = df.loc[mask_fit, unknown_variables]

    Xp = df.loc[mask_predict, known_variables]
    # yp = df.loc[mask_predict, unknown_variables]
    fh = ForecastingHorizon(values=Xp.index, is_relative=False)

    yp = forecaster.fit_predict(y=yf, X=Xf, fh=fh, X_pred=Xp)

    df1 = df.copy()
    df1.loc[mask_predict, unknown_variables] = yp
    return df1, forecaster
