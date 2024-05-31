from scipy.stats import loguniform, uniform
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sktime.forecasting.compose import ForecastingPipeline, TransformedTargetForecaster
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.trend import TrendForecaster
from sktime.transformations.series.adapt import TabularToSeriesAdaptor


def get_default_estimators(Tin: int) -> list[Pipeline]:
    """Returns a default set of estimators that will be used if the user did not
    specify a specific list of estimators to use."""
    tscv = TimeSeriesSplit(n_splits=Tin)

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
        "kernel_ridge__gamma": uniform(0.5 * 1 / 50, 2 * 1 / 50),
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
        # pipeline_naive,
        # pipeline_linear_regression,
        pipeline_kernel_ridge_cv,
        pipeline_svr_cv,
    ]

    pipe_y_naive = TransformedTargetForecaster(
        steps=[
            ("scaler", TabularToSeriesAdaptor(StandardScaler())),
            (
                "forecaster",
                NaiveForecaster(strategy="last"),
            ),
        ]
    )

    pipe_X_naive = ForecastingPipeline(
        steps=[
            ("scaler", TabularToSeriesAdaptor(StandardScaler())),
            ("naive", pipe_y_naive),
        ]
    )

    pipe_y_linear_regression = TransformedTargetForecaster(
        steps=[
            ("scaler", TabularToSeriesAdaptor(StandardScaler())),
            (
                "forecaster",
                TrendForecaster(regressor=LinearRegression(fit_intercept=False)),
            ),
        ]
    )

    pipe_X_linear_regression = ForecastingPipeline(
        steps=[
            ("scaler", TabularToSeriesAdaptor(StandardScaler())),
            ("PCA", TabularToSeriesAdaptor(PCA(n_components=0.9))),
            ("forecaster", pipe_y_linear_regression),
        ]
    )

    model_list = [
        pipe_X_naive,
        pipe_X_linear_regression,
    ]

    return model_list
