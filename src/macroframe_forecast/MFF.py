# Disclaimer: Reuse of this tool and IMF information does not imply
# any endorsement of the research and/or product. Any research presented
# should not be reported as representing the views of the IMF,
# its Executive Board, member governments.


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

# %% MFF


class MFF:
    """A class for Macro-Framework Forecasting (MFF).

    This class facilitates forecasting of single frequency time series data
    using a two-step process. First step of the forecasting procedure generates
    unconstrained forecasts using the forecaster specified. In the next step,
    these forecasts are then reconclied so that they satisfy the supplied
    constrants, and smoothness of the forecasts is maintained.

    Parameters
    ----------
    df : pd.DataFrame
       Input dataframe containing time series data. Data should be in wide
       format, with each row containing data for one period, and each
       column containing data for one variable.

    forecaster : BaseForecaster, optional(default: None)
        sktime BaseForecaster descendant. If not defined, then DefaultForecaster
        is used.

    constraints_with_wildcard : str, optional(default: None)
        Constraints that hold with equality. Constraints may include wildcard,
        in which case constraints will be applied across all horizons, or
        may be defined for specified time periods.

    ineq_constraints_with_wildcard : str, optional(default: None)
        Inequality constraints, comparable to ``constraints_with_wildcard``.
        Constraints may include wildcard, in which case constraints will be
        applied across all horizons, or may be defined for specified time
        periods. Constraints should be written in the form of 'C_ineq*y - d_ineq ≤ 0 '. 

    parallelize : boolean
        Indicate whether parallelization should be employed for generating the
        first step forecasts. Default value is `True`.

    n_forecast_error : int
        Number of windows to split data into training and testing sets for
        generating matrix of forecast errors. Default is 5.

    shrinkage_method : str, optional(default: 'oas')
        Method to be used for shrinking sample covariance matrix. Default is
        Oracle Shrinking Approximating Estimator ('oas'). Other options are
        oas, identity and monotone_diagonal.

    default_lam : float, optional(default: -1)
        The value of lambda to be used for calculating smoothing parameter if
        frequency of observations cannot be determined from index names. If this
        is set to -1, lambda is calculated empirically. Default is -1.

    max_lam : float, optional(default: 129600)
        Maximum value of lamstar to be used for smoothing forecasts when being
        estimated empirically.

    Returns
    -------
    df2 : pd.Dataframe
        Output dataframe with all reconciled forecasts filled into the original
        input.


    """

    def __init__(
        self,
        df: pd.DataFrame,
        forecaster: BaseForecaster | None = None,
        equality_constraints: list[str] = [],
        inequality_constraints: list[str] = [],
        parallelize: bool = True,
        n_forecast_error: int = 5,
        shrinkage_method: str = "oas",
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
        """
        Fits the model and generates reconciled forecasts for the input
        dataframe subject to defined constraints.
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
