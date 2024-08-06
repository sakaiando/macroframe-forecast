from scipy.stats import loguniform, uniform
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.compose import (
    DirectReductionForecaster,
    ForecastingPipeline,
    MultiplexForecaster,
    TransformedTargetForecaster,
)
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.forecasting.naive import NaiveForecaster
from sktime.split import ExpandingGreedySplitter, SlidingWindowSplitter
from sktime.transformations.series.adapt import TabularToSeriesAdaptor


def get_default_forecaster(Tin: int, window_length: int = 1) -> BaseForecaster:
    """Returns a default forecaster (grid search with cross validation) that will
    be used if the user did not specify a specific list of estimators to use.
    :param window_length: """

    pipe_y_naive = TransformedTargetForecaster(
        steps=[
            ("scaler", TabularToSeriesAdaptor(StandardScaler())),
            (
                "forecaster",
                NaiveForecaster(strategy="drift"),
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
                DirectReductionForecaster(
                    estimator=LinearRegression(fit_intercept=False), window_length=window_length
                ),
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

    # elastic net

    pipe_y_elastic_net = TransformedTargetForecaster(
        steps=[
            ("scaler", TabularToSeriesAdaptor(StandardScaler())),
            (
                "forecaster",
                DirectReductionForecaster(
                    estimator=ElasticNetCV(fit_intercept=False, max_iter=500), window_length=window_length
                ),
            ),
        ]
    )

    pipe_X_elastic_net = ForecastingPipeline(
        steps=[
            ("scaler", TabularToSeriesAdaptor(StandardScaler())),
            ("forecaster", pipe_y_elastic_net),
        ]
    )

    # kernel ridge

    pipe_y_kernel_ridge = TransformedTargetForecaster(
        steps=[
            ("scaler", TabularToSeriesAdaptor(StandardScaler())),
            (
                "forecaster",
                DirectReductionForecaster(
                    estimator=KernelRidge(kernel="rbf"), window_length=window_length
                ),
            ),
        ]
    )

    pipe_X_kernel_ridge = ForecastingPipeline(
        steps=[
            ("scaler", TabularToSeriesAdaptor(StandardScaler())),
            ("forecaster", pipe_y_kernel_ridge),
        ]
    )

    gridsearch_cv_kernel_ridge = ForecastingGridSearchCV(
        forecaster=pipe_X_kernel_ridge,
        param_grid=[
            {
                "forecaster__forecaster__alpha": loguniform(0.1, 1000).rvs(size=3, random_state=0),
                "forecaster__forecaster__gamma": uniform(0.5 * 1 / 50, 2 * 1 / 50).rvs(
                    size=3, random_state=0
                ),
            },
        ],
        cv=SlidingWindowSplitter(window_length=10),
    )

    # forecaster representation for selection among the listed models
    forecaster = MultiplexForecaster(
        forecasters=[
            ("naive", pipe_X_naive),
#            ("linear_reg", pipe_X_linear_regression),
#            ("elasticnet", pipe_X_elastic_net),
#            ("kernel_ridge", gridsearch_cv_kernel_ridge),
        ],
    )

    cv = ExpandingGreedySplitter(test_size=4, folds=3)  # TODO: allow more customization here

    # choose among the provided forecasters
    gscv = ForecastingGridSearchCV(
        forecaster=forecaster,
        cv=cv,
        param_grid={
            "selected_forecaster": [
                "naive",
#                "linear_reg",
#                "elasticnet",
#                "kernel_ridge",
            ],
        },
        tune_by_variable=True,
        backend='dask'
    )

    return gscv
