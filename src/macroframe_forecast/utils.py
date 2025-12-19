# Disclaimer: Reuse of this tool and IMF information does not imply
# any endorsement  of the research and/or product. Any research presented
# should not be reported as representing the views of the IMF,
# its Executive Board, member governments.

import copy
import re
import warnings
from random import sample, seed
from string import ascii_lowercase
from time import time
from typing import Literal

import cvxpy as cp
import numpy as np
import pandas as pd
import scipy
import sympy as sp
from dask.distributed import Client
from numpy import ndarray
from numpy.linalg import inv
from pandas import DataFrame, Index, PeriodIndex, Series
from scipy.linalg import block_diag
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.model_selection import TimeSeriesSplit
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
from sktime.split import ExpandingGreedySplitter
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sktime.transformations.series.feature_selection import FeatureSelection

# %%


def CheckTrainingSampleSize(df0: DataFrame, n_forecast_error: int = 5) -> bool:
    """
    Check sample size available for training window. Raise an exception if the
    number of observations available is too low.

    Parameters
    ----------

    df0 : pd.DataFrame
        Input dataframe with island values replaced by nan.

    n_forecast_error : int
        Number of training and testing sets to split data into for generating
        matrix of forecast errors.

    Returns
    -------

    small_sample : bool
        Indicator for whether the sample of observations available for training
        is small.

    """

    forecast_horizon = max(np.argwhere(df0.isna())[:, 0]) - min(np.argwhere(df0.isna())[:, 0]) + 1

    minimum_training_obs = min(np.argwhere(df0.isna())[:, 0]) - forecast_horizon - n_forecast_error

    if minimum_training_obs <= 0:
        raise ValueError(
            "Number of observations too low for given forecast horizon "
            "and n_sample_splits; consider reducing forecast horizon and/or "
            "n_sample_splits"
        )

    elif minimum_training_obs <= 15:
        return True

    else:
        return False


def DefaultForecaster(small_sample: bool = False) -> BaseForecaster:
    """
    Set up forecasting pipeline, specifying the scaling (transforming) to be
    applied and forecasting model to be used.

    Parameters
    ----------
    small_sample : boolean
        Indicator for whether the sample of observations available for training
        is small. By default this is turned to False.

    Returns
    -------
    gscv : BaseForecaster
        Instance of sktime's Grid Search forecaster, derived from BaseForecaster,
        which is configured for hyperparameter tuning and model selection.

    """

    pipe_y_elasticnet = TransformedTargetForecaster(
        steps=[
            ("scaler", TabularToSeriesAdaptor(StandardScaler())),
            ("forecaster", DirectReductionForecaster(ElasticNetCV(max_iter=5000,
                                                                  cv=TimeSeriesSplit(n_splits=5)),
                                                     window_length = 5)),
        ]
    )

    pipe_yX_elasticnet = ForecastingPipeline(
        steps=[
            ("scaler", TabularToSeriesAdaptor(StandardScaler())),
            ("pipe_y", pipe_y_elasticnet),
        ]
    )

    ols_1feature = ForecastingPipeline(
        steps=[
            ("feature_selection", FeatureSelection(n_columns=1)),
            ("ols", DirectReductionForecaster(LinearRegression())),
        ]
    )

    ols_pca = ForecastingPipeline(
        steps=[
            ("pca", TabularToSeriesAdaptor(PCA(n_components=0.9))),
            ("ols", DirectReductionForecaster(LinearRegression())),
        ]
    )

    # forecaster representation for selection among the listed models
    forecaster = MultiplexForecaster(
        forecasters=[
            ("naive_drift", NaiveForecaster(strategy="drift", window_length=2)),
            ("naive_last", NaiveForecaster(strategy="last")),
            ("naive_mean", NaiveForecaster(strategy="mean", window_length=5)),
            ("elasticnetcv", pipe_yX_elasticnet),
            ("ols_1feature", ols_1feature),
            ("ols_pca", ols_pca),
        ]
    )

    cv = ExpandingGreedySplitter(test_size=1, folds=5)

    # If the number of observations is small, Grid Search is no longer used for
    # model selection. Instead, OLS with PCA is used is used.

    if not small_sample:
        gscv = ForecastingGridSearchCV(
            forecaster=forecaster,
            cv=cv,
            param_grid={
                "selected_forecaster": [
                    "naive_drift",
                    "naive_last",
                    "naive_mean",
                    "elasticnetcv",
                    "ols_1feature",
                    "ols_pca",
                ]
            },
            backend=None,
        )

    else:
        gscv = NaiveForecaster(strategy = "last")

    return gscv


def CleanIslands(df: DataFrame) -> tuple[DataFrame, Series]:
    """
    Separate island values from input dataframe, replacing them with nan.
    Called by ``OrganizeCells``.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with raw data.

    Returns
    -------
    df_no_islands : pd.DataFrame
        Dataframe with island values replaced by nan.

    islands : pd.Series
        Series containing island values.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> n = 30
        >>> p = 2
        >>> df = pd.DataFrame(np.random.sample([n,p]),
        >>>                   columns=['a','b'],
        >>>                   index=pd.date_range(start='2000',periods=n,freq='YE').year)
        >>> df.iloc[-5:-1,:1] = np.nan
        >>> df0, islands = CleanIslands(df)

    """
    df_no_islands = df.copy()  # to keep original df as it is
    col_with_islands = df.columns[df.isna().any()]
    coli_list = [df_no_islands.columns.get_loc(col) for col in col_with_islands]
    for coli in coli_list:  # for col with na
        first_na_index = np.argwhere(df.iloc[:, coli].isna()).min()
        df_no_islands.iloc[first_na_index:, coli] = np.nan

    islands: Series = df[df_no_islands.isna()].T.stack()
    return df_no_islands, islands


def OrganizeCells(df: DataFrame) -> tuple[DataFrame, Series, Series, Series]:
    """
    Extract island values (if existing) from input dataframe, replacing them
    with nan values. This is useful for generating first step forecasts, which
    disregard known island values for the prediction. Also identifies separate
    Pandas series of names of cells for known and unknown values in the input
    dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with raw data.

    Returns
    -------
    df0 : pd.DataFrame
        Dataframe with island values replaced by nan.

    all_cells : pd.Series
        Series containing cell names of all cells in the input dataframe.

    unknown_cells : pd.Series
        Series containing cell names of cells whose values are to be forecasted.

    known_cells : pd.Series
        Series containing cell names of cells whose values are known.

    islands : pd.Series
        Series containing island values.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> n = 30
    >>> p = 2
    >>> df = pd.DataFrame(np.random.sample([n,p]),
    >>>                   columns=['a','b'],
    >>>                   index=pd.date_range(start='2000',periods=n,freq='YE').year)
    >>> df.iloc[-5:-1,:1] = np.nan
    >>> df0, all_cells, unknown_cells, known_cells, islands = OrganizeCells(df)
    """

    # clean islands
    df0, islands = CleanIslands(df)

    # all cells in forecast horizon
    all_cells_index = df0.T.stack(future_stack=True).index
    all_cells = pd.Series([f"{a}_{b}" for a, b in all_cells_index], index=all_cells_index)

    # unknown cells with nan
    unknown_cells_index = df0.isna()[df0.isna()].T.stack().index
    unknown_cells = pd.Series([f"{a}_{b}" for a, b in unknown_cells_index], index=unknown_cells_index)

    # known cells
    known_cells_index = all_cells_index.difference(unknown_cells_index)
    known_cells = pd.Series([f"{a}_{b}" for a, b in known_cells_index], index=known_cells_index)

    return df0, all_cells, unknown_cells, known_cells, islands


def find_permissible_wildcard(constraints_with_wildcard: list[str], _seed: int = 0) -> str:
    """Generate random letter to be used in constraints."""
    wild_card_length = 1
    seed(_seed)
    candidate = "".join(sample(ascii_lowercase, wild_card_length))
    while candidate in "".join(constraints_with_wildcard):
        wild_card_length = wild_card_length + 1
        candidate = "".join(sample(ascii_lowercase, wild_card_length))
    alphabet_wildcard = candidate
    return alphabet_wildcard


def find_strings_to_replace_wildcard(constraint: str, var_list: Series, wildcard: str) -> list[str]:
    """Identify list of strings to be substituted with the wildcard character."""

    varlist_regex = ["^" + str(v).replace(wildcard, "(.*)") + "$" for v in sp.sympify(constraint).free_symbols]
    missing_string_set_list = []
    for w in varlist_regex:
        missing_string = []
        for v in var_list:
            match = re.compile(w).search(v)
            if match:
                missing_string.append(match.group(1))
        missing_string_set_list.append(set(missing_string))
    missing_string_list = list(set.intersection(*missing_string_set_list))
    missing_string_list.sort()

    return missing_string_list


def expand_wildcard(constraints_with_alphabet_wildcard: list[str], var_list: Series, wildcard: str):
    """
    Expand constraints with wildcard to all possible time periods. This is
    called within ``StringToMatrixConstraints``, and the wildcard character
    has already been replaced by a random letter before this function is
    called.

    Parameters
    ----------
    constraints_with_alphabet_wildcard : string
        Linear equality constraints with wildcard string replaced
        with alphabets.
    var_list : list
        List of indices of all cells (known and unknown) in raw dataframe.
    wildcard : string
        Alphabet which has replaced wildcard string in the constraints.

    Return
    ------
    expanded_constraints : list
        Expanded list of constraints over all time periods.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> n = 30
    >>> p = 2
    >>> df = pd.DataFrame(np.random.sample([n,p]),
    >>>                   columns=['a','b'],
    >>>                   index=pd.date_range(start='2000',periods=n,freq='YE').year)
    >>> df0_stacked = df.T.stack()
    >>> all_cells_index = df0_stacked.index
    >>> var_list = pd.Series([f'{a}_{b}' for a, b in all_cells_index],
    >>>                      index = all_cells_index)
    >>> constraints_with_alphabet_wildcard = ['ax + bx']
    >>> alphabet_wildcard = 'x'
    >>> constraints = expand_wildcard(constraints_with_alphabet_wildcard,
    >>>                               var_list = var_list,
    >>>                               wildcard = alphabet_wildcard)

    """
    expanded_constraints = []
    for constraint in constraints_with_alphabet_wildcard:
        if wildcard not in constraint:
            expanded_constraints.append(constraint)
        else:
            missing_string_list = find_strings_to_replace_wildcard(constraint, var_list, wildcard)
            expanded_constraints += [constraint.replace(f"{wildcard}", m) for m in missing_string_list]
    return expanded_constraints


def StringToMatrixConstraints(
    df0_stacked: DataFrame,  # stack df0 to accomodate mixed frequency
    all_cells: Series,
    unknown_cells: Series,
    known_cells: Series,
    constraints_with_wildcard: list[str] | None = None,
    wildcard_string: str = "?",
) -> tuple[DataFrame, DataFrame]:
    """
    Convert equality constraints from list to matrix form for horizons to
    be forecasted (Cy = d, where C and d are dataframes containing the
    linear constraints). The input dataframe should not be in a standard wide
    format, but instead all columns should be stacked on one another. This is
    needed to control for dealing with the case of mixed frequency among
    observations. All island values in the dinput dataframe should be replaced
    by nan prior to this step.

    Parameters
    ----------
    df0_stacked : pd.Series
        Stacked version of df0 (input  dataframe with islands removed).
    all_cells : pd.Series
        Series containing cell names of all cells in the input dataframe.
    unknown_cells : pd.Series
        Series containing cell names of cells whose values are to be forecasted.
    known_cells : pd.Series
        Series containing cell names of cells whose values are known..
    constraints_with_wildcard : str, optional
        String specifying equality constraints that have to hold.
        The default is [].
    wildcard_string : str, optional
        String that is used as wildcard identifier in constraint.
        The default is '?'.

    Returns
    -------
    C: pd.DataFrame
        Dataframe containing matrix of the linear constraints on the left side of
        equation Cy=d.
    d: pd.DataFrame
        Dataframe containing matrix of the linear constraints on the right side of
        equation Cy=d.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> n = 30
    >>> p = 2
    >>> df = pd.DataFrame(np.random.sample([n,p]),
    >>>                   columns=['a','b'],
    >>>                   index=pd.date_range(start='2000',periods=n,freq='YE').year)
    >>> df.iloc[-5:-1,:1] = np.nan
    >>> df0, all_cells, unknown_cells, known_cells, islands = OrganizeCells(df)
    >>> df0_stacked = df0.T.stack()
    >>> constraints_with_wildcard = ['a?+b?']
    >>> C,d = StringToMatrixConstraints(df0_stacked,
    >>>                                 all_cells,
    >>>                                 unknown_cells,
    >>>                                 known_cells,
    >>>                                 constraints_with_wildcard)
    """

    if constraints_with_wildcard is None:
        constraints_with_wildcard = list()

    # replace wildcard with alphabet to utilize sympy
    alphabet_wildcard = find_permissible_wildcard(constraints_with_wildcard)
    constraints_with_alphabet_wildcard = [
        c.replace(wildcard_string, alphabet_wildcard) for c in constraints_with_wildcard
    ]

    # expand constraints using all cells at forecast horizon
    constraints = expand_wildcard(
        constraints_with_alphabet_wildcard, var_list=all_cells.tolist(), wildcard=alphabet_wildcard
    )

    # obtain C_unknown by differentiating constraints wrt unknown cells with nan
    A, b = sp.linear_eq_to_matrix(constraints, sp.sympify(unknown_cells.tolist()))
    C = pd.DataFrame(np.array(A).astype(float), index=constraints, columns=unknown_cells.index)
    nonzero_rows = (C != 0).any(axis=1)
    C = C.loc[nonzero_rows]  # drop rows with all zeros

    # obtain d_unknown by substituting known cells
    known_cell_dict = pd.Series(
        [df0_stacked.loc[idx] for idx in known_cells.index], index=known_cells.tolist()
    ).to_dict()
    d = pd.DataFrame(np.array(b.subs(known_cell_dict)).astype(float), index=constraints)
    d = d.loc[nonzero_rows]  # drop rows with all zeros in C

    return C, d


def AddIslandsToConstraints(C: DataFrame, d: DataFrame, islands: Series) -> tuple[DataFrame, DataFrame]:
    """
    Add island values into the matrix form equality constraints which have been
    constructed by ``StringToMatrixConstraints``.

    Parameters
    ----------
    C : pd.DataFrame
        Dataframe containing matrix of the linear constraints on the left side of
        equation Cy=d.
    d : pd.DataFrame
        Dataframe containing matrix of the linear constraints on the right side of
        equation Cy=d.
    islands : pd.Series
        Series containing island values to be introduced into linear equation.

    Returns
    -------
    C_aug : pd.DataFrame
        Dataframe containing the augmented C matrix, with island values incorporated.
    d_aug : pd.DataFrame
        Dataframe containing the augmented d vector, with island values incorporated.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> n = 30
    >>> p = 2
    >>> df = pd.DataFrame(np.random.sample([n,p]),
    >>>                   columns=['a','b'],
    >>>                   index=pd.date_range(start='2000',periods=n,freq='YE').year)
    >>> df.iloc[-5:-1,:1] = np.nan
    >>> df0, all_cells, unknown_cells, known_cells, islands = OrganizeCells(df)
    >>> df0_stacked = df0.T.stack()
    >>> constraints_with_wildcard = ['a?+b?']
    >>> C,d = StringToMatrixConstraints(df0_stacked,
    >>>                                 all_cells,
    >>>                                 unknown_cells,
    >>>                                 known_cells,
    >>>                                 constraints_with_wildcard)
    >>> C,d = AddIslandsToConstraints(C,d,islands)
    """
    C_aug_index = islands.index.union(C.index, sort=False)  # singleton constraints prioritize over islands
    C_aug = pd.DataFrame(np.zeros([len(C_aug_index), len(C.columns)]), index=C_aug_index, columns=C.columns)
    d_aug = pd.DataFrame(np.zeros([len(C_aug_index), 1]), index=C_aug_index)
    for idx in islands.index:
        C_aug.loc[C_aug.index == idx, idx] = 1
        d_aug.loc[d_aug.index == idx] = islands.loc[idx]
    C_aug.update(C)
    d_aug.update(d)

    return C_aug, d_aug


def FillAnEmptyCell(
    df: DataFrame, row: int | str, col: int | str, forecaster: BaseForecaster
) -> tuple[float, BaseForecaster]:
    """
    Generate a forecast for a given cell based on the latest known value
    for the given column (variable) and using the predefined forecasting pipeline.
    Called by ``FillAllEmptyCells``.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing known values of all variables and nan for
        unknown values.
    row : str
        Row index of cell to be forecasted.
    col : str
        Column index of cell to be forecasted.
    forecaster : BaseForecaster


    Returns
    -------
    y_pred : double
        Forecasted value of the variable for the given horizon.
    forecaster : BaseForecaster
         sktime BaseForecaster descendant

    Examples
    --------
    >>> from string import ascii_lowercase
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.linear_model import ElasticNetCV
    >>> from sktime.forecasting.compose import YfromX
    >>> n = 30
    >>> p = 2
    >>> df = pd.DataFrame(np.random.sample([n,p]),
    >>>                   columns=list(ascii_lowercase[:p]),
    >>>                   index=pd.date_range(start='2000',periods=n,freq='YE').year)
    >>> df.iloc[-5:,:1] = np.nan
    >>> row = df.index[-1]
    >>> col = df.columns[0]
    >>> forecaster = YfromX(ElasticNetCV())
    >>> y_pred, forecaster = FillAnEmptyCell(df,row,col,forecaster)
    """
    warnings.filterwarnings("ignore", category=UserWarning)

    # clone a forecaster
    f = forecaster.clone()
    
    # last historical data and forecast horizon in num
    T = np.argwhere(df.loc[:, col].isna()).min() - 1
    h = np.where(df.index == row)[0][0] - T

    y = df.iloc[:T, :].loc[:, col]

    X = df.iloc[: T + h].drop(columns=[col]).dropna(axis=1)
    X_train = X.iloc[:T, :]
    X_pred = X.iloc[T:, :]

    y_pred = f.fit(y=y, X=X_train, fh=h).predict(X=X_pred)

    return y_pred, f

def FillAllEmptyCells(
    df: DataFrame, forecaster: BaseForecaster, parallelize: bool = True
) -> tuple[DataFrame, DataFrame]:
    """
    Generate forecasts for all unknown cells in the supplied dataframe.
    All forecasts are made independently from each other. (TBC)

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe containing known values of all variables and nan for
        unknown values.

    forecaster : BaseForecaster
        sktime BaseForecaster descendant

    parallelize : boolean
        Indicate whether parallelization should be employed for generating the
        first step forecasts. Default value is `True`.

    Return
    ------
    df1: pd.DataFrame
        Dataframe with all known cells, as well as unknown cells filled in by
        one-step forecasts.
    df1_model: pd.DataFrame
        Dataframe with all known cells, with unknown cells containing details
        of the forecaster used for generating forecast of that cell.

    Examples
    --------
    >>> from string import ascii_lowercase
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.linear_model import ElasticNetCV
    >>> from sktime.forecasting.compose import YfromX
    >>> from mff.utils import FillAllEmptyCells
    >>> n = 30
    >>> p = 2
    >>> df = pd.DataFrame(np.random.sample([n,p]),
    >>>                   columns=list(ascii_lowercase[:p]),
    >>>                   index=pd.date_range(start='2000',periods=n,freq='YE').year)
    >>> df.iloc[-5:,:1] = np.nan
    >>> def DefaultForecaster():
    >>>     return YfromX(ElasticNetCV(max_iter=5000))
    >>> df1,df1_models = FillAllEmptyCells(df,DefaultForecaster())

    """

    # get indices of all np.nan cells
    na_cells = [(df.index[rowi], df.columns[coli]) for rowi, coli in np.argwhere(df.isna())]

    # apply dask
    if parallelize:
        start = time()
        client = Client()
        df_future = client.scatter(df,broadcast=True)
        forecaster_future = client.scatter(forecaster, broadcast=True)
        futures = [client.submit(FillAnEmptyCell, df_future, row, col, forecaster_future)
                   for (row, col) in na_cells]
        results = client.gather(futures)
        client.close()
        print("Dask filled", len(results), "out-of-sample cells:", round(time() - start, 3), "seconds")

    else:
        start = time()
        results = [FillAnEmptyCell(df, row, col, forecaster) for row, col in na_cells]
        print("Forecast", len(results), "cells:", round(time() - start, 3), "seconds")

    # fill empty cells
    df1 = df.copy()
    df1_models = df.copy().astype(object)
    for idx, rowcol in enumerate(na_cells):
        df1.loc[rowcol] = results[idx][0].iloc[0]
        df1_models.loc[rowcol] = results[idx][1]

    return df1, df1_models


def GenPredTrueData(
    df: DataFrame, forecaster: BaseForecaster, n_forecast_error: int = 5, parallelize: bool = True
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """
    Generate in-sample forecasts from existing data by constructing
    pseudo-historical datasets.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with all known as well as unknown values.
    forecaster : BaseForecaster
        sktime BaseForecaster descendant.
    n_forecast_error : int, optional
        Number of horizons for which in-sample forecasts are generated.
        The default is 5.
    parallelize : boolean, optional
        Indicate whether parallelization should be used. The default is True.

    Returns
    -------
    pred : pd.DataFrame
        Dataframe with in-sample predictions generated using pseudo-historical
        datasets.
    true : pd.DataFrame
        Dataframe with actual values of the variable corresponding to predicted
        values contained in pred.
    model : pd.DataFrame
        Dataframe with information on the models used for generating each
        forecast.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sktime.forecasting.compose import YfromX
    >>> from sklearn.linear_model import ElasticNetCV
    >>> n = 30
    >>> p = 2
    >>> df = pd.DataFrame(np.random.sample([n,p]),
    >>>                   columns=['a','b'],
    >>>                   index=pd.date_range(start='2000',periods=n,freq='YE').year)
    >>> df.iloc[-5:,:1] = np.nan
    >>> def DefaultForecaster():
    >>>     return YfromX(ElasticNetCV(max_iter=5000))
    >>> pred,true,model = GenPredTrueData(df0,forecaster,parallelize=parallelize)
    """

    # last historical data and length of forecast horizon
    T = min(np.argwhere(df.isna())[:, 0]) - 1
    h = max(np.argwhere(df.isna())[:, 0]) - T

    # create pseudo historical dataframes and their na cells
    df_list = [df.shift(-h - n).mask(df.shift(-h - n).notna(), df).iloc[: -h - n, :] for n in range(n_forecast_error)]

    # unpack all the na cells for pseudo historical dataframes to use dask
    tasks = [
        (dfi, df.index[rowi], df.columns[coli])
        for dfi, df in enumerate(df_list)
        for (rowi, coli) in np.argwhere(df.isna())
    ]

    if parallelize:
        start = time()
        client = Client()
        df_futures = client.scatter(df_list, broadcast=True)
        forecaster_future = client.scatter(forecaster, broadcast=True)
        futures = [client.submit(FillAnEmptyCell, df_futures[dfi], row, col, forecaster_future) for (dfi, row, col) in tasks]
        results = client.gather(futures)
        client.close()
        print("Dask filled", len(results), "in-sample cells:", round(time() - start, 3), "seconds")
    else:
        start = time()
        results = [FillAnEmptyCell(df_list[dfi], row, col, forecaster) for (dfi, row, col) in tasks]
        print("Fill", len(results), "in-sample cells:", round(time() - start, 3), "seconds")

    # repackage results by filling na of df_list
    filled_list = copy.deepcopy(df_list)
    model_list = [df.astype(object) for df in copy.deepcopy(df_list)]
    for task_idx, task in enumerate(tasks):
        dfi, row, col = task
        filled_list[dfi].loc[row, col] = results[task_idx][0].iloc[0]
        model_list[dfi].loc[row, col] = results[task_idx][1]

    # reduce n samples into a dataframe
    colname = df.isna()[df.isna()].T.stack().index
    idxname = pd.Index(
        [df_list[n].index[np.argwhere(df_list[n].isna())[:, 0].min()] for n in range(n_forecast_error)], name="LastData"
    )
    pred = pd.DataFrame(
        [filled_list[n][df_list[n].isna()].T.stack().values for n in range(n_forecast_error)],
        index=idxname,
        columns=colname,
    )
    model = pd.DataFrame(
        [model_list[n][df_list[n].isna()].T.stack().values for n in range(n_forecast_error)],
        index=idxname,
        columns=colname,
    )
    true = pd.DataFrame(
        [df[df_list[n].isna()].T.stack().values for n in range(n_forecast_error)], index=idxname, columns=colname
    )

    return pred, true, model


def BreakDataFrameIntoTimeSeriesList(
    df0: DataFrame, df1: DataFrame, pred: DataFrame, true: DataFrame
) -> tuple[list[DataFrame], list[DataFrame], list[DataFrame]]:
    """Transform relevant dataframes into lists for ensuing reconciliation step.

    Parameters
    ----------
    df0 : pd.DataFrame
        Dataframe with all known and unknown values, without any islands.
    df1 : pd.DataFrame
        Dataframe with unknown values as well as islands filled in with
        first step forecasts.
    pred : pd.DataFrame
        Dataframe with in-sample predictions generated using pseudo-historical
        datasets, output from ``GenPredTrueData``.
    true : pd.DataFrame
       Dataframe with actual values of the variable corresponding to predicted
        values contained in pred.

    Returns
    -------
    ts_list : list
        List containing all first step out of sample forecasts.
    pred_list : list
        List of dataframes, with each dataframe containing in-sample forecasts
        for one variable.
    true_list : list
        List of dataframes, with each dataframe containing the actual values
        for a variable corresponding to in-sample predictions stored in
        pred_list.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sktime.forecasting.compose import YfromX
    >>> from sklearn.linear_model import ElasticNetCV
    >>> n = 30
    >>> p = 2
    >>> df = pd.DataFrame(np.random.sample([n,p]),
    >>>                   columns=['a','b'],
    >>>                   index=pd.date_range(start='2000',periods=n,freq='YE').year)
    >>> df.iloc[-5:,:1] = np.nan
    >>> def DefaultForecaster():
    >>>     return YfromX(ElasticNetCV(max_iter=5000))
    >>> df1,df1_models = FillAllEmptyCells(df,DefaultForecaster())
    >>> pred,true,model = GenPredTrueData(df0,forecaster,parallelize=parallelize)
    >>> ts_list,pred_list,true_list = BreakDataFrameIntoTimeSeriesList(df,df1,pred,true)
    """
    ts_list = [df1[df0.isna()].loc[:, col:col].dropna().T.stack() for col in df0.columns[df0.isna().any()]]
    pred_list = [pred.loc[:, ts.index] for ts in ts_list]
    true_list = [true.loc[:, ts.index] for ts in ts_list]

    return ts_list, pred_list, true_list


def HP_matrix(size: int) -> ndarray:
    """
    Create the degenerate penta-diagonal matrix (the one used in HP Filter),
    with dimensions (size x size).

    Parameters
    ----------
    size : integer
        Number of rows for the square matrix.

    Returns
    -------
    F : np.array
        Array containing the F matrix.

    """
    if size >= 2:
        D = np.zeros((size - 2, size))
        for i in range(size - 2):
            D[i, i] = 1
            D[i, i + 1] = -2
            D[i, i + 2] = 1
        F = D.T @ D
    elif size == 1:
        F = np.zeros([1, 1])
    return F


def GenVecForecastWithIslands(ts_list: list[DataFrame], islands: list[Series]) -> Series:
    """Overwrite forecasted values for islands with known island value.

    Parameters
    ----------
    ts_list : list
        List of all first step forecasted values.
    islands : pd.Series
        Series containing island values.

    Returns
    -------
    y1 : pd.Series
        Series of forecasted values with island values incorporated.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sktime.forecasting.compose import YfromX
    >>> from sklearn.linear_model import ElasticNetCV
    >>> n = 30
    >>> p = 2
    >>> df = pd.DataFrame(np.random.sample([n,p]),
    >>>                   columns=['a','b'],
    >>>                   index=pd.date_range(start='2000',periods=n,freq='YE').year)
    >>> df.iloc[-5:-1,:1] = np.nan
    >>> df0, all_cells, unknown_cells, known_cells, islands = OrganizeCells(df)
    >>> def DefaultForecaster():
    >>>     return YfromX(ElasticNetCV(max_iter=5000))
    >>> df1,df1_models = FillAllEmptyCells(df,DefaultForecaster(),parallelize=False)
    >>> ts_list = [df1[df0.isna()].loc[:,col:col].dropna().T.stack() for col in df0.columns[df.isna().any()]]
    >>> y1 = GenVecForecastWithIslands(ts_list,islands)
    """
    try:
        y1 = pd.concat(ts_list, axis=0)

    except Exception:  # only used in mixed-freq, pd.concat cann't process 4 mix-freq series
        y1 = ConcatMixFreqMultiIndexSeries(ts_list, axis=0)

    y1.update(islands)

    return y1


def GenWeightMatrix(
    pred_list: list[DataFrame], true_list: list[DataFrame], shrinkage_method: Literal["oas", "oasd"] = "oas"
) -> tuple[DataFrame, float]:
    """
    Generate weighting matrix based on in-sample forecasts and actual values
    for the corresponding periods.

    Parameters
    ----------
    pred_list : list
        List of dataframes, with each dataframe containing in-sample forecasts
        for one variable..
    true_list : list
        List of dataframes, with each dataframe containing the actual values
        for a variable corresponding to in-sample predictions stored in
        pred_list.
    shrinkage_method : str, optional
        Type of algorithm to use for shrinking the covariance matrix, with
        options of identity, oas and oasd. The default is 'oas'.

    Returns
    -------
    W : pd.DataFrame
        Weighting matrix to be used for reconciliation.
    shrinkage: float
        Shrinkage parameter associated with the weight. Nan in case identity
        is selected as method.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> pred_list = [pd.DataFrame(np.random.rand(5, 5), columns=[f'Col{i+1}' for i in range(5)]) for _ in range(2)]
    >>> true_list = [pd.DataFrame(np.random.rand(5, 5), columns=[f'Col{i+1}' for i in range(5)]) for _ in range(2)]
    >>> W,shrinkage = GenWeightMatrix(pred_list, true_list)

    """
    fe_list = [pred_list[i] - true_list[i] for i in range(len(pred_list))]

    try:  # fe: sample size x vairables
        fe = pd.concat(fe_list, axis=1)

    except Exception:  # only used in mixed-freq, pd.concat cann't process 4 mix-freq series
        fe = ConcatMixFreqMultiIndexSeries(fe_list, axis=1)

    # sample covariance
    n_samp = fe.shape[0]
    n_vars = fe.shape[1]
    sample_cov = fe.cov()

    if shrinkage_method == "identity":
        W = pd.DataFrame(np.eye(sample_cov.shape[0]), index=sample_cov.index, columns=sample_cov.columns)
        return W, np.nan

    if shrinkage_method == "oas":
        from sklearn.covariance import OAS

        oas = OAS().fit(fe.values)
        W = pd.DataFrame(oas.covariance_, index=sample_cov.index, columns=sample_cov.columns)
        rho = oas.shrinkage_
        return W, rho

    if shrinkage_method == "oasd":
        if n_vars >= 2:
            # shrinkage target
            diag = np.diag(np.diag(sample_cov))

            # shrinkage parameter
            numerator = np.trace(sample_cov @ sample_cov) - np.trace(diag @ diag)
            denominator = np.trace(sample_cov @ sample_cov) + np.trace(sample_cov) ** 2 - 2 * np.trace(diag @ diag)
            phi = numerator / denominator
            rho = min([1 / (n_samp * phi), 1])

            # shrink covariance matrix
            W = (1 - rho) * sample_cov + rho * diag
        elif n_vars == 1:
            W = sample_cov
            rho = np.nan
        return W, rho

    if shrinkage_method == "monotone diagonal":
        if n_vars >= 2:
            diag = pd.Series(np.diag(sample_cov), index=sample_cov.index)
            W = pd.DataFrame(
                np.diag(diag.groupby(level=0).cummax()), index=sample_cov.index, columns=sample_cov.columns
            )
        elif n_vars == 1:
            W = sample_cov
            rho = np.nan
        return W, np.nan


def GenLamstar(pred_list: list, true_list: list, default_lam: float = -1, max_lam: float = 129600) -> pd.Series:
    """
    Calculate the smoothness parameter (lambda) associated with each variable
    being forecasted.

    Parameters
    ----------
    pred_list : list
        List of dataframes, with each dataframe containing in-sample forecasts
        for one variable.
    true_list : list
        List of dataframes, with each dataframe containing the actual values
        for a variable corresponding to in-sample predictions stored in
        pred_list.
    default_lam : float, optional(default: -1)
        The value of lambda to be used for calculating smoothing parameter if
        frequency of observations cannot be determined from index names. If this
        is set to -1, lambda is calculated empirically. The default value is -1.
    max_lam : float, optional
        The upperbound of HP filter penalty term (lambda) searched by scipy
        minimizer. The default is 129600.

    Returns
    -------
    lamstar : pd.Series
        Series containing smoothing parameters to be used for each variable.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> pred_list = [pd.DataFrame(np.random.rand(5, 5), columns=[f'Col{i+1}' for i in range(5)]) for _ in range(2)]
    >>> true_list = [pd.DataFrame(np.random.rand(5, 5), columns=[f'Col{i+1}' for i in range(5)]) for _ in range(2)]
    >>> W,shrinkage = GenWeightMatrix(pred_list, true_list)
    """
    # index of time series to deal with mixed-frequency
    tsidx_list = [df.columns for df in pred_list]

    # box to store lamstar, columsn are the index of time series
    try:  # extract freq info if available
        freq_list = [tsidx.get_level_values(1).freqstr[0] for tsidx in tsidx_list]
        ly = 100
        lambda_dict = {
            "Y": ly,
            "Q": ly * (4**2),
            "M": ly * (12**2),
            "W": ly * (52**2),
            "D": ly * (365**2),
            "H": ly * ((365 * 24) ** 2),
            "T": ly * ((365 * 24 * 60) ** 2),
            "S": ly * ((365 * 24 * 60 * 60) ** 2),
        }
        lamstar = pd.Series([float(lambda_dict[item]) for item in freq_list], index=tsidx_list)
    except Exception:
        lamstar = pd.Series(np.ones(len(tsidx_list)) * default_lam, index=tsidx_list)

    # optimal lambda
    if default_lam == -1:

        def loss_fn(x, T, yt, yp):
            return (yt - inv(np.eye(T) + x * HP_matrix(T)) @ yp).T @ (yt - inv(np.eye(T) + x * HP_matrix(T)) @ yp)

        for tsidxi, tsidx in enumerate(tsidx_list):
            y_pred = pred_list[tsidxi]
            y_true = true_list[tsidxi]
            T = len(tsidx)

            # TODO: pick a better name for the function
            def obj(x):
                return np.mean(
                    [
                        loss_fn(x, T, y_true.iloc[i : i + 1, :].T.values, y_pred.iloc[i : i + 1, :].T.values)
                        for i in range(y_pred.shape[0])
                    ]
                )

            constraint_lb = {"type": "ineq", "fun": lambda lam: lam}  # lambda >=0

            # lambda <= max_lam, without this, I+xF may be too close to F to invert
            constraint_ub = {"type": "ineq", "fun": lambda lam: -lam + max_lam}
            result = scipy.optimize.minimize(obj, 0, constraints=[constraint_lb, constraint_ub])
            lamstar.iloc[tsidxi] = result.x[0]
    return lamstar


def GenSmoothingMatrix(W: DataFrame, lamstar: Series) -> DataFrame:
    """
    Generate symmetric smoothing matrix using optimal lambda and weighting matrix.

    Parameters
    ----------
    W : pd.DataFrame
        Dataframe containing the weighting matrix.
    lamstar : pd.Series
        Series containing smoothing parameters to be used for each variable.

    Returns
    -------
    Phi : pd.DataFrame
        Dataframe containing the smoothing matrix.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> pred_list_1 = [pd.DataFrame(np.random.rand(5, 5),
    >>>                             columns=pd.MultiIndex.from_product([['A'], [f'Col{i+1}' for i in range(5)]])) if i == 0 else
    >>>                pd.DataFrame(np.random.rand(5, 5),
    >>>                             columns=pd.MultiIndex.from_product([['B'], [f'Col{i+1}' for i in range(5)]]))
    >>>                for i in range(2)]
    >>> true_list_1 = [pd.DataFrame(np.random.rand(5, 5),
    >>>                             columns=pd.MultiIndex.from_product([['A'], [f'Col{i+1}' for i in range(5)]])) if i == 0 else
    >>>                pd.DataFrame(np.random.rand(5, 5),
    >>>                             columns=pd.MultiIndex.from_product([['B'], [f'Col{i+1}' for i in range(5)]]))
    >>>                for i in range(2)]
    >>> smoothness = GenLamstar(pred_list_1,true_list_1)

    """
    lam = lamstar / [np.diag(W.loc[tsidx, tsidx]).min() for tsidx in lamstar.index]
    Phi_np = block_diag(*[lam.iloc[tsidxi] * HP_matrix(len(tsidx)) for tsidxi, tsidx in enumerate(lam.index)])
    Phi = pd.DataFrame(Phi_np, index=W.index, columns=W.columns)
    return Phi


def Reconciliation(
    y1: Series,
    W: DataFrame,
    Phi: DataFrame,
    C: DataFrame,
    d: DataFrame,
    C_ineq: DataFrame | None = None,
    d_ineq: DataFrame | None = None,
) -> DataFrame:
    """
    Reconcile first step forecasts to satisfy equality as well as inequality
    constraints, subject to smoothening.

    Parameters
    ----------
    y1 : pd.Series
        Series of all forecasted and island values.
    W : pd.DataFrame
        Dataframe containing the weighting matrix.
    Phi : pd.DataFrame
        Dataframe containing the smoothing matrix.
    C : pd.DataFrame
        Dataframe containing matrix of the linear constraints on the left side of
        the equality constraint Cy=d.
    d : pd.DataFrame
        Dataframe containing matrix of the linear constraints on the right side of
        the equality constraint Cy=d.
    C_ineq : pd.DataFrame, optional
        Dataframe containing matrix of the linear constraints on the left side of
        the inequality constraint C_ineq · y - d_ineq ≤ 0. The default is None.
    d_ineq : pd.DataFrame, optional
        Dataframe containing matrix of the linear constraints on the right side of 
        the inequality constraint C_ineq · y - d_ineq ≤ 0.  The default is None.

    Returns
    -------
    y2 : pd.DataFrame
        Dataframe containing the final reconciled forecasts for all variables.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sktime.forecasting.compose import YfromX
    >>> from sklearn.linear_model import ElasticNetCV
    >>> n = 30
    >>> p = 2
    >>> df = pd.DataFrame(np.random.sample([n,p]),
    >>>                   columns=['a','b'],
    >>>                   index=pd.date_range(start='2000',periods=n,freq='YE').year)
    >>> df.iloc[-5:,:1] = np.nan
    >>> df0, all_cells, unknown_cells, known_cells, islands = OrganizeCells(df)
    >>> def DefaultForecaster():
    >>>     return YfromX(ElasticNetCV(max_iter=5000))
    >>> df1,df1_models = FillAllEmptyCells(df0,DefaultForecaster(),parallelize=False)
    >>> pred,true,model = GenPredTrueData(df0,DefaultForecaster(),parallelize=False)
    >>> ts_list,pred_list,true_list = BreakDataFrameIntoTimeSeriesList(df0,df1,pred,true)
    >>> y1 = pd.concat(ts_list)
    >>> C = pd.DataFrame(columns = y1.index).astype(float)
    >>> d = pd.DataFrame().astype(float)
    >>> W = pd.DataFrame(np.eye(5),index=y1.index,columns=y1.index)
    >>> smoothness = GenLamstar(pred_list,true_list)
    >>> Phi = GenSmoothingMatrix(W,smoothness)
    >>> y2 = Reconciliation(y1,W,Phi,C,d)
    >>> y2 = Reconciliation(m.y1,m.W,m.Phi,m.C,m.d)

    """
    assert (y1.index == W.index).all()
    assert (y1.index == Phi.index).all()
    assert (y1.index == C.columns).all()
    assert (C.index == d.index).all()

    def DropLinDepRows(C_aug, d_aug):
        C = C_aug.values

        # Convert the matrix to a SymPy Matrix
        sympy_matrix = sp.Matrix(C)

        # Compute the RREF and get the indices of linearly independent rows
        rref_matrix, independent_rows = sympy_matrix.T.rref()

        # Extract the independent rows
        independent_rows = list(independent_rows)

        # dependent rows
        all_rows = set(range(C.shape[0]))
        dependent_rows = list(all_rows - set(independent_rows))

        C = C_aug.iloc[independent_rows, :]
        d = d_aug.iloc[independent_rows, :]

        if dependent_rows != []:
            print(
                "Constraints are linearly dependent. The following constraints are dropped.",
                C_aug.index[dependent_rows],
            )
        return C, d

    # keep lin indep rows
    C, d = DropLinDepRows(C, d)

    # reconcile with np.array
    W_inv = inv(W)
    denom = inv(W_inv + Phi)
    Cn = C.values
    dn = d.values
    CdC_inv = inv(Cn @ denom @ Cn.T)  # removing linearly dependent rows to use inv doesn't change results much

    In = np.eye(len(y1))
    y1n = y1.values.reshape(-1, 1)
    y2n = (In - denom @ Cn.T @ CdC_inv @ Cn) @ denom @ W_inv @ y1n + denom @ Cn.T @ CdC_inv @ dn

    if C_ineq is not None and C_ineq.shape[0] > 0:
        C_ineq, d_ineq = DropLinDepRows(C_ineq, d_ineq)

        # augment C_ineq, d_ineq to be compatible with y1
        C_ineq_aug = pd.DataFrame(np.zeros([len(C_ineq.index), len(y1)]), index=C_ineq.index, columns=y1.index)
        C_ineq_aug.update(C_ineq)
        d_ineq_aug = pd.DataFrame(np.zeros([len(d_ineq.index), 1]), index=d_ineq.index)
        d_ineq_aug.update(d_ineq)
        Cn_ineq = C_ineq_aug.values
        dn_ineq = d_ineq_aug.values

        # use CVXPY to solve numerically
        P = W_inv + Phi
        q = -2 * W_inv @ y1n
        x = cp.Variable([len(y1), 1])
        objective = cp.Minimize(cp.quad_form(x, P, assume_PSD=True) + q.T @ x)
        
        # If equality constraints do not exist, dropping C matrix from solver
        if C.shape[0] >0:
            constraints = [Cn @ x == dn, Cn_ineq @ x <= dn_ineq]
        else:
            constraints = [Cn_ineq @ x <= dn_ineq]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        y2n = x.value

        if y2n is None:
            import warnings

            warnings.warn("Reconciliation failed. Feasible sets might be empty.")

    # put reconciled y2 back to dataframe
    y2 = pd.DataFrame(y2n, index=y1.index)

    return y2


def get_freq_of_freq(periodindex: PeriodIndex, freqstr: Literal["Y", "Q", "M", "W", "D", "H", "T", "S"]) -> Index:
    if freqstr == "Y":
        return periodindex.year
    if freqstr == "Q":
        return periodindex.quarter
    if freqstr == "M":
        return periodindex.month
    if freqstr == "W":
        return periodindex.week
    if freqstr == "D":
        return periodindex.day
    if freqstr == "H":
        return periodindex.hour
    if freqstr == "T":
        return periodindex.minute
    if freqstr == "S":
        return periodindex.second


def ConcatMixFreqMultiIndexSeries(df_list: list[DataFrame], axis: int) -> DataFrame:
    # used only in mixed freq case, pd.concat doesn't work for more than 4 mix-freq series
    # doesn't work when there are more than 3 freq!
    try:
        return pd.concat(df_list, axis=axis)
    except Exception:
        if axis == 0:
            # concat by freq
            freqs = [df.index.get_level_values(1).freqstr[0] for df in df_list]
            seen = set()
            freq_unique = [x for x in freqs if not (x in seen or seen.add(x))]
            dflong_list = []
            for k in freq_unique:
                df_list_k = [df for df in df_list if df.index.get_level_values(1).freqstr[0] == k]
                dflong_k = pd.concat(df_list_k, axis=0)
                dflong_list.append(dflong_k)

            dflong = pd.concat(dflong_list, axis=0)
            return dflong

        if axis == 1:
            # concat by freq
            freqs = [df.columns.get_level_values(1).freqstr[0] for df in df_list]
            seen = set()
            freq_unique = [x for x in freqs if not (x in seen or seen.add(x))]
            dfwide_list = []
            for k in freq_unique:
                df_list_k = [df for df in df_list if df.columns.get_level_values(1).freqstr[0] == k]
                dfwide_k = pd.concat(df_list_k, axis=1)
                dfwide_list.append(dfwide_k)

            dfwide = pd.concat(dfwide_list, axis=1)
            return dfwide
