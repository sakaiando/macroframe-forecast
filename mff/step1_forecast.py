import numpy as np
import pandas as pd
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon


class UnconditionalForecaster:
    def __init__(self, df, forecaster, forecast_start=None, forecast_end=None):

        self.df = df
        self.df_no_islands = self.delete_exogenous_islands(forecast_start)
        self.forecaster_base = forecaster
        self.forecast_start = self.find_forecast_start_by_col(df) if forecast_start is None else forecast_start
        self.forecast_end = self.df.index.max() if forecast_end is None else forecast_end

        self.n_horizons = self.forecast_end - self.forecast_start
        self.is_staggered = not len(self.forecast_start.unique()) <= 2  # 2 dates for exogenous and endogenous cols

        self.forecaster_trained = None
        self.fh = None
        self.fcast_dict = None
        self.y_hat = None
        self.cov_mat = None
        self.oos_resids = None

    def calculate_forecast_interval(self):
        ss = self.df.index[self.df.isna().any(axis=1)]
        return ss.get_level_values('year').min(), ss.get_level_values('year').max

    def delete_exogenous_islands(self, forecast_start):
        if forecast_start is None:
            return self.delete_exogenous_islands_no_start()
        else:
            return self.delete_exogenous_islands_with_start()

    def delete_exogenous_islands_no_start(self):
        df = self.df.copy()
        last_data = self.find_forecast_start_by_col(df)
        for col in last_data.index:
            df.loc[last_data[col]:, col] = np.nan
        return df

    def delete_exogenous_islands_with_start(self):
        df = self.df.copy()
        endog_cols = df.isna().any()
        df.loc[self.forecast_start, endog_cols] = np.nan
        return df

    @staticmethod
    def find_forecast_start_by_col(df):
        return df.isna().sort_index().idxmax(axis=0).replace({df.index.min(): df.index.max() + 1})

    def fit(self):
        df = self.df_no_islands.copy()
        df = df.T.reset_index(drop=True).T  # sktime will not handle multiindex columns
        step_dates = self.forecast_start.unique() - 1  # TODO: -1 won't work with datetime, but will work with PeriodIndex and int
        step_dates.sort()
        if len(step_dates) > 1:  # if no staggered or exog
            step_dates = step_dates[1:]

        fcast_dict = {}
        fh_dict = {}
        y_hat = df.copy()
        for step in step_dates:
            y_hat_temp, fh_dict[step], fcast_dict[step] = self.unconstrained_forecast(df.loc[:step])
            y_hat.update(y_hat_temp, overwrite=False)
        self.y_hat, self.fh, self.forecaster_trained = y_hat, fh_dict, fcast_dict
        self.y_hat.columns = self.df.columns

    def unconstrained_forecast(self,
                               df: pd.DataFrame,
                               fh: ForecastingHorizon or int = None
                               ) -> tuple[pd.DataFrame | ForecastingHorizon | BaseForecaster]:
        forecaster = self.forecaster_base  # TODO: does this need to be copied?
        # if df.isna().any().sum() < df.shape[1] or (df.isna().sum().sum() == 0):
        #     yf = df
        #     Xf = None
        #     Xp = None
        # else:
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
            df1.update(yp, overwrite=False)  # doesn't replace filled values

        return df1, fh, forecaster

    def fit_covariance(self,
                       n_periods):  # TODO: could construct a covariance class here instead of putting in the UnconstraintedForecaster object
        self.oos_resids = self.calculate_oos_residuals(n_periods)
        self.cov_mat = self.oos_resids.cov()

    def calculate_oos_residuals(self, n_periods):
        df = self.df_no_islands
        resids = []
        n_horizons = self.n_horizons.max()
        for h in range(n_horizons, n_horizons + n_periods):
            h = h + 1
            df_est = df[~df.shift(-h).isna()].dropna(how='all')
            pred = UnconditionalForecaster(df_est, self.forecaster_base)
            pred.fit()
            resid = (df - pred.y_hat)[df_est.isna()].shift(h)
            resid.index = resid.index
            resid = resid.T.stack().dropna()
            resids.append(resid)
        resids = pd.concat(resids, axis=1).T
        return resids

    @staticmethod
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
    import pandas as pd
    from mff.mff.default_forecaster import get_default_forecaster
    from mff.mff.utils import load_synthetic_data

    df, _ = load_synthetic_data()
    forecaster = get_default_forecaster(10)
    forecaster = UnconditionalForecaster(df, forecaster)
    forecaster.fit()