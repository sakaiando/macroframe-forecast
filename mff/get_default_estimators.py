from scipy.stats import loguniform, uniform
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sktime.forecasting.compose import ForecastingPipeline, TransformedTargetForecaster
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.trend import TrendForecaster
from sktime.split import SlidingWindowSplitter
from sktime.transformations.series.adapt import TabularToSeriesAdaptor


def get_default_estimators(Tin: int) -> list[ForecastingPipeline | ForecastingGridSearchCV]:
    """Returns a default set of estimators that will be used if the user did not
    specify a specific list of estimators to use."""

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

    # elastic net

    pipe_y_elastic_net = TransformedTargetForecaster(
        steps=[
            ("scaler", TabularToSeriesAdaptor(StandardScaler())),
            (
                "forecaster",
                TrendForecaster(regressor=ElasticNetCV(fit_intercept=False, max_iter=500)),
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
                TrendForecaster(regressor=KernelRidge(kernel="rbf")),
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

    model_list = [
        pipe_X_naive,
        pipe_X_linear_regression,
        pipe_X_elastic_net,
        gridsearch_cv_kernel_ridge,
    ]

    return model_list
