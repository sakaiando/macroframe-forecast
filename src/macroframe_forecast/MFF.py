# Disclaimer: Reuse of this tool and IMF information does not imply
# any endorsement of the research and/or product. Any research presented
# should not be reported as representing the views of the IMF,
# its Executive Board, member governments.


from typing import Literal

import pandas as pd
from sktime.forecasting.base import BaseForecaster

from macroframe_forecast.utils import (
    AddIslandsToConstraints,
    BreakDataFrameIntoTimeSeriesList,
    CheckTrainingSampleSize,
    DefaultForecaster,
    FillAllEmptyCells,
    GenLamstar,
    GenPredTrueData,
    GenSmoothingMatrix,
    GenVecForecastWithIslands,
    GenWeightMatrix,
    OrganizeCells,
    Reconciliation,
    StringToMatrixConstraints,
)


class MFF:
    """A class for Macro-Framework Forecasting (MFF).

    Forecast single-frequency time series data by generating initial forecasts
    and reconciling them to satisfy the supplied constraints while maintaining
    forecast smoothness.

    Parameters
    ----------
    df : pd.DataFrame
        Input data in wide format, with periods in rows and variables in
        columns. Missing values indicate cells to forecast.

    forecaster : BaseForecaster, optional
        An ``sktime`` forecaster. If omitted, the package's default forecaster
        is used.

    equality_constraints : list[str], optional
        Linear equality constraints, expressed as strings that equal zero.
        Use ``?`` as a wildcard to apply a constraint across forecast periods.
        Defaults to an empty list.

    inequality_constraints : list[str], optional
        Linear inequality constraints, expressed in the form ``expression <= 0``.
        Use ``?`` as a wildcard to apply a constraint across forecast periods.
        Defaults to an empty list.

    parallelize : bool, optional
        Whether to parallelize generation of the initial forecasts. Defaults
        to ``True``.

    n_forecast_error : int, optional
        Number of pseudo-out-of-sample forecast windows used to estimate
        forecast errors. Defaults to ``5``.

    shrinkage_method : {"oas", "identity", "oasd", "monotone diagonal"}, optional
        Method used to shrink the sample covariance matrix. Defaults to
        ``"oas"``.

    default_lam : float, optional
        Smoothing parameter used when observation frequency cannot be inferred
        from the index. A value of ``-1`` estimates it empirically. Defaults to
        ``-1``.

    max_lam : float, optional
        Maximum smoothing parameter considered when estimating it empirically.
        Defaults to ``129600``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        forecaster: BaseForecaster | None = None,
        equality_constraints: list[str] = [],
        inequality_constraints: list[str] = [],
        parallelize: bool = True,
        n_forecast_error: int = 5,
        shrinkage_method: Literal["oas", "identity", "oasd", "monotone diagonal"] = "oas",
        default_lam: float = -1,
        max_lam: float = 129600
    ):
        self.df = df
        self.forecaster = forecaster
        self.equality_constraints = equality_constraints
        self.inequality_constraints = inequality_constraints
        self.parallelize = parallelize
        self.n_forecast_error = n_forecast_error
        self.shrinkage_method = shrinkage_method
        self.default_lam = default_lam
        self.max_lam = max_lam

    def fit(
        self,
    ) -> pd.DataFrame:
        """Fit the model and return forecasts reconciled to the defined constraints.

        Returns
        -------
        pd.DataFrame
            A copy of the input data with forecasted values filled in and
            reconciled to satisfy the constraints.
        """

        df = self.df
        forecaster = self.forecaster
        equality_constraints = self.equality_constraints
        inequality_constraints = self.inequality_constraints
        parallelize = self.parallelize
        n_forecast_error = self.n_forecast_error
        shrinkage_method = self.shrinkage_method
        default_lam = self.default_lam
        max_lam = self.max_lam

        # modify inputs into machine-friendly shape
        df0, all_cells, unknown_cells, known_cells, islands = OrganizeCells(df)

        # get constraint matrices
        C, d = StringToMatrixConstraints(df0.T.stack(), all_cells, unknown_cells, known_cells, equality_constraints)
        C, d = AddIslandsToConstraints(C, d, islands)
        C_ineq, d_ineq = StringToMatrixConstraints(
            df0.T.stack(), all_cells, unknown_cells, known_cells, inequality_constraints
        )

        # Initiate DefaultForecaster only if a forecaster has not already been defined by the user.
        # Use OLS PCA if small_sample is True, and Grid Search if false.
        small_sample: bool = CheckTrainingSampleSize(df0, n_forecast_error)
        if forecaster is None:
            forecaster = DefaultForecaster(small_sample)
        
        # 1st stage forecast and its model
        df1, df1_model = FillAllEmptyCells(df0, forecaster, parallelize=parallelize)

        # get pseudo out-of-sample prediction, true values, and prediction models
        pred, true, model = GenPredTrueData(df0, forecaster, n_forecast_error=n_forecast_error, parallelize=parallelize)

        # break dataframe into list of time series
        ts_list, pred_list, true_list = BreakDataFrameIntoTimeSeriesList(df0, df1, pred, true)

        # get parts for reconciliation
        y1 = GenVecForecastWithIslands(ts_list, islands)
        W, shrinkage = GenWeightMatrix(pred_list, true_list, shrinkage_method=shrinkage_method)
        smoothness = GenLamstar(pred_list, true_list, default_lam=default_lam, max_lam=max_lam)
        Phi = GenSmoothingMatrix(W, smoothness)

        # 2nd stage reconciled forecast
        y2 = Reconciliation(y1, W, Phi, C, d, C_ineq, d_ineq)

        # reshape vector y2 into df2
        y2 = y2.T.stack(future_stack=True)
        y2.index = y2.index.droplevel(level=0)
        df2 = df0.copy()
        df2.update(y2, overwrite=False)  # fill only nan cells of df0

        self.df0 = df0
        self.C = C
        self.d = d
        self.C_ineq = C_ineq
        self.d_ineq = d_ineq
        self.islands = islands

        self.df1 = df1
        self.df1_model = df1_model

        self.pred = pred
        self.true = true
        self.model = model
        self.ts_list = ts_list
        self.pred_list = pred_list
        self.true_list = true_list
        self.y1 = y1
        self.W = W
        self.Phi = Phi
        self.shrinkage = shrinkage
        self.smoothness = smoothness

        self.y2 = y2
        self.df2 = df2

        return self.df2
