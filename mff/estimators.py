from typing import Self

import numpy as np
from joblib import parallel_backend
from numpy import array, full
from pandas import DataFrame, concat
from scipy.stats import loguniform, uniform
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def augment_lag(df, lag):
    """
    augment_lag adds lags of df

    Parameters
     ----------
     df: dataframe
     lag: int
         the number of lags used as regressors in the step1 training

    Returns
     -------
     dfaug: dataframe
         If df is n x m, dfaug is (n-lag) x (m x lag)
    """
    df_list = [df]
    for Li in range(1, lag + 1):
        Ld = df.shift(Li)
        Ld.columns = ["L" + str(Li) + "_" + str(vn) for vn in df.columns]
        df_list.append(Ld)
    dfaug = concat(df_list, axis=1).iloc[lag:, :]
    return dfaug


class NaiveForecaster(BaseEstimator, RegressorMixin):
    """Naive forecaster to return the last value as prediction."""

    def fit(self, X, y) -> Self:
        self.last_value_ = y[-1]
        return self

    def predict(self, X) -> array:
        return full(shape=(len(X),), fill_value=self.last_value_)


def estimate_relationship(df: DataFrame, lag: int, Tin: int) -> tuple[DataFrame]:
    # Augment lags
    df0aug = augment_lag(df, lag)  # more columns, fewer rows
    print("Original df shape", df.shape)
    print("Augmented df shape", df0aug.shape)

    # extract information on T,h,u,k from the shape of df0
    T_aug = sum(~np.isnan(df0aug).any(axis=1))  # length of historical data
    h = len(df0aug) - T_aug  # length of forecast horizon
    m_aug = df0aug.shape[1]  # number of all variables
    k_aug = sum(
        ~np.isnan(df0aug.iloc[: T_aug + 1, :]).any(axis=0)
    )  # number of known variables in T+1 including lags
    u = m_aug - k_aug  # m-k = m_aug - k_aug # number of unknown variables
    print(f"{T_aug=} {h=} {m_aug=} {k_aug=} {u=}")

    # create sub-dataframe and their np versions
    df0aug_u = df0aug.iloc[:, :u]  # not df0_u since rows are different from df0
    df0aug_k = df0aug.iloc[:, u:]

    # drop columns with missing values (due to lags)
    cols_nans = df0aug_k.isna().sum(axis=0) == 0
    df0aug_k = df0aug_k.loc[:, cols_nans].copy()

    df0aug_u_np = df0aug_u.to_numpy()
    df0aug_k_np = df0aug_k.to_numpy()

    # Step1 Prediction for T+1
    df0aug_h = df0aug.copy()  # hat, will be reshaped to df1 later
    df0aug_fitted_model = {}  # storage for fitted model
    df0aug_h_regularization = df0aug.copy()  # regularization = Elastic Net CV
    df0aug_h_dim_reduction = df0aug.copy()  # dimension reduction = Principal Component
    df0aug_h_naive = df0aug.copy()  # regularization = Elastic Net CV
    df0aug_h_kernel_ridge = df0aug.copy()  # kernel ridge
    df0aug_h_svr = df0aug.copy()  # support vector regression
    df0aug_fitted_model_regularization = DataFrame(
        index=df0aug_u.index, columns=df0aug.columns
    )  # storage for ElasticNet fit
    df0aug_fitted_model_dim_reduction = DataFrame(
        index=df0aug_u.index, columns=df0aug.columns
    )  # storage for ols + pca fit
    df0aug_fitted_model_kernel_ridge = DataFrame(
        index=df0aug_u.index, columns=df0aug.columns
    )  # storage for ols + pca fit
    df0aug_fitted_model_svr = DataFrame(
        index=df0aug_u.index, columns=df0aug.columns
    )  # storage for ols + pca fit

    tscv = TimeSeriesSplit(n_splits=Tin)

    pipeline_linear_regression: Pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("PCA", PCA(n_components=0.9)),
            ("linreg", LinearRegression(fit_intercept=False)),
        ]
    )

    pipeline_naive = Pipeline(
        [
            ("naive", NaiveForecaster()),
        ]
    )

    pipeline_elastic_net = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "elasticnet",
                ElasticNetCV(
                    cv=tscv,
                    max_iter=500,
                    fit_intercept=False,
                ),
            ),
        ]
    )

    pipeline_kernel_ridge = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("kernel_ridge", KernelRidge(kernel="rbf")),
        ]
    )

    param_distributions = {
        "kernel_ridge__alpha": loguniform(0.1, 1000),
        "kernel_ridge__gamma": uniform(0.5 * 1 / df.shape[1], 2 * 1 / df.shape[1]),
    }

    pipeline_kernel_ridge_cv = Pipeline(
        [
            (
                "randomsearch_cv_kernel",
                RandomizedSearchCV(
                    pipeline_kernel_ridge,
                    param_distributions=param_distributions,
                    n_iter=500,
                    random_state=0,
                    cv=tscv,
                ),
            )
        ]
    )

    pipeline_svr = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf")),
        ]
    )

    param_distributions = {
        "svr__C": loguniform(0.1, 1000),
    }

    pipeline_svr_cv = Pipeline(
        [
            (
                "randomsearch_cv_svr",
                RandomizedSearchCV(
                    pipeline_svr,
                    param_distributions=param_distributions,
                    n_iter=500,
                    random_state=0,
                    cv=tscv,
                ),
            )
        ]
    )

    model_list: list[Pipeline] = [
        pipeline_elastic_net,
        pipeline_naive,
        pipeline_linear_regression,
        pipeline_kernel_ridge_cv,
        pipeline_svr_cv,
    ]

    # unknown_variables: list[str] = ['var1', 'var2', '...']

    with parallel_backend(backend="loky"):
        # for uvar in unknown_variables:
        for ui in list(range(u)):
            performance_across_models = dict()
            for model_num, model in enumerate(model_list):
                transformed_model = TransformedTargetRegressor(
                    regressor=model, transformer=StandardScaler()
                )

                y_forecasts = []
                y_true_vals = []
                for t in range(
                    T_aug - Tin, T_aug + 1
                ):  # forecast of T-Tin to T is for the weight matrix in the 2nd step
                    X = df0aug_k_np[:t, :]
                    y = df0aug_u_np[:t, ui].reshape(-1, 1)

                    transformed_model.fit(X, y)

                    X_pred = df0aug_k_np[t, :].reshape(1, -1)
                    y_true = df0aug_u_np[t, ui].reshape(1, -1)

                    y_est = transformed_model.predict(X_pred)

                    y_forecasts.append(y_est.reshape(1, -1)[0][0])
                    y_true_vals.append(y_true[0])

                forecast_error = mean_absolute_error(y_true_vals[:-1], y_forecasts[:-1])
                performance_across_models[model_num] = {
                    "forecast_error": forecast_error,
                    "fit_model": transformed_model,
                    "predicted_values": y_forecasts,
                }

                print(f"For variable {ui},  model {transformed_model} has score: {forecast_error}")

            # select the best model based on forecast error (lower is better, see sklearn.metrics)
            best_model_number = min(
                performance_across_models,
                key=lambda item: performance_across_models[item]["forecast_error"],
            )
            best_model = performance_across_models[best_model_number]["fit_model"]

            print(f"For variable {ui} the best model is {best_model} with score: {forecast_error}")

            # fill-in the dataframe with best predictions for each variable
            df0aug_h.iloc[T_aug - Tin : T_aug + 1, ui] = performance_across_models[
                best_model_number
            ]["predicted_values"]

            # forecast of T+2 to T+h, if h = 1 nothing will happen
            for t in range(-h + 1, 0):
                # drop lag variables and re-augment
                df0_h = df.copy()
                df0_h.iloc[-h - Tin :, :u] = df0aug_h.iloc[-h - Tin :, :u]
                df0aug_h = augment_lag(df0_h, lag)
                df0aug_k = df0aug_h.iloc[:, u:]

                # drop columns with missing values (due to lags)
                df0aug_k = df0aug_k.loc[:, cols_nans].copy()

                X_pred = df0aug_k.iloc[t, :].values.reshape(1, -1)

                y_est = best_model.predict(X_pred)

                df0aug_h.iloc[t, ui] = y_est

    # drop lags and add the rows that were dropped when lags are augmented
    df1 = df.copy()
    df1.iloc[-h - Tin :, :u] = df0aug_h.iloc[-h - Tin :, :u]

    # reorder variables to match df0
    df1 = df1[df.columns]

    return df1, df0aug_fitted_model
