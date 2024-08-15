import numpy as np
import pandas as pd
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon


class UnreconciledForecaster:
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
        self.cov = None
        self.oos_resids = None

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
        forecaster = self.forecaster_base
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
        fh = ForecastingHorizon(values=Xp.index + 1, is_relative=True)
        yp = forecaster.fit_predict(y=yf, X=Xf, fh=fh, X_pred=Xp)

        df1 = df.copy()
        if Xp is None:
            df1 = pd.concat([df1, yp])
        else:
            df1.update(yp, overwrite=False)  # doesn't replace filled values

        return df1, fh, forecaster

    def fit_covariance(self,
                       n_periods, how):
        self.oos_resids = self.calculate_oos_residuals(n_periods)
        self.cov = CovarianceMatrix(self.oos_resids, how)

    def calculate_oos_residuals(self, n_periods):
        df = self.df_no_islands
        resids = []
        n_horizons = self.n_horizons.max()
        for h in range(n_horizons, n_horizons + n_periods):
            h = h + 1
            df_est = df[~df.shift(-h).isna()].dropna(how='all')
            pred = UnreconciledForecaster(df_est, self.forecaster_base)
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

    def get_best_forecaster(self):
        best_params = {}
        for k, v in self.forecaster_trained.items():
            if hasattr(v, 'best_params_'):
                model = v.best_params_['selected_forecaster']
            else:
                model = v.forecasters_.applymap(lambda x: x.best_params_['selected_forecaster'])
            best_params[k] = model
        return best_params


class CovarianceMatrix:
    def __init__(self, resids, how):
        self._how = how
        self.resids = resids
        self.cov_mat = self.calc_covariance()

    def raw_covariance(self):
        return self.resids.cov()

    def oasd_covariance(self):
        sig_hat = self.raw_covariance()

        def phi():
            diag = np.diag(np.diag(sig_hat))
            numerator = np.trace(sig_hat @ sig_hat) - np.trace(diag @ diag)
            denom = np.trace(sig_hat @ sig_hat) + np.trace(sig_hat) ** 2 - 2 * np.trace(diag @ diag)
            return numerator / denom

        rho = min(1 / (len(self.resids) * phi()), 1)

        return rho * sig_hat + (1 - rho) * np.diag(np.diag(sig_hat))

    def calc_covariance(self):
        calculator = {'oasd': self.oasd_covariance,
                      'raw': self.raw_covariance,
                      'montone_diagonal': self.monotone_diag_covariance}
        return calculator[self.how]()

    def monotone_diag_covariance(self):
        sig_hat = self.raw_covariance()

        sig_hat_diags = np.diag(sig_hat)
        sig_hat_diags = pd.Series(sig_hat_diags, index=sig_hat.index
                                  ).sort_index(level=['variable', 'freq', 'subperiod', 'year']
                                               ).groupby(['variable', 'freq', 'subperiod']
                                                         ).cummax()
        sig_hat_adj = pd.DataFrame(np.diag(sig_hat_diags), index=sig_hat_diags.index, columns=sig_hat_diags.index).replace({0: np.nan})
        sig_hat_adj.update(sig_hat, overwrite=False)
        return sig_hat_adj

    @property
    def how(self):
        return self._how

    @how.setter
    def how(self, how):
        self._how = how
        self.cov_mat = self.calc_covariance()


if __name__ == '__main__':
    import pandas as pd
    from mff.mff.default_forecaster import get_default_forecaster
    from mff.mff.utils import load_synthetic_data

    df, _ = load_synthetic_data()
    forecaster = get_default_forecaster(10)
    forecaster = UnreconciledForecaster(df, forecaster)
    forecaster.fit()
    forecaster.fit_covariance(2, 'monotone_diagonal')
    forecaster.cov.how = 'oasd'

