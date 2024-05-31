from typing import Optional

from numpy import array
from numpy.linalg import matrix_rank
from pandas import DataFrame, to_numeric
from sklearn.pipeline import Pipeline

from mff.estimators import unconstrained_forecast
from mff.reconciliation import forecast_reconciliation
from mff.validators import can_forecast, is_consistent_shape


def constrained_forecast(
    df: DataFrame,
    lag: int,
    Tin: int,
    C_dict: dict,
    d_dict: dict,
    estimators: Optional[Pipeline | list[Pipeline]] = None,
) -> tuple[DataFrame, DataFrame, None]:
    """
    Parameters
    ----------
    df: dataframe
        (T+h) x m dataframe representing input data
        the first m-k columns of T:T+h rows are nan, and the rest are not nan
        if the columns are not sorted in this order, a sorted version df0 will
        be produced
    lag: int
        the number of lags used as regressors in the step1 training
        if all variables are unknown, lag should be > 0
    Tin: int
        the number of time periods in historical data used to estimate forecast-error
    C_dict: dictionary
        T+h length dictionary, each element of which is
        (m-n) x m numpy matrix C of floag type in the constraint C x df.columns = d
        the order of columns must be the same of the columns of df
        n is the number of free variables (net contents)
    d_dict: dictionary
        T+h length dictionary of numpy array, each element of which is
        (m-n) x 1 column vector in the constraint C x df = d

    Returns
    -------
    df2: dataframe
        (T+h) x m dataframe with nan values filled using the two-step forecasting method
    df1: dataframe
        (T+h) x m dataframe with nan values filled using the first step of the forecasting method
    df0aug_fitted_model: dictionary
        u + 2 length dictionary, each element of which is a fit object,
        containing the estimated cofficients for an unknown variable.
        The first u keys are unknown variables, and the last two store
        coefficients from regularization method and dimension reduction method
    """

    df = DataFrame(df)  # make sure df is dataframe
    df = df.apply(to_numeric)  # make sure df elements are numeric

    # check1: make sure there are something to forecas
    if not can_forecast(df):
        raise ValueError("Nothing to forecast. Extend time for some variables.")

    # check 2: make sure the size of contraint matrix is consistent with number of vars
    if not is_consistent_shape(df, constraints=C_dict):
        raise ValueError("The size of contraint matrix is not consistent with number of vars.")

    # check3: check whether unknown variables come before known variables, if not, change the columns of C in C_dict
    u_var = df.columns[df.isna().any()].tolist()
    k_var = df.columns[~df.isna().any()].tolist()
    correct_order = u_var + k_var
    if sum(df.columns != correct_order):
        df0 = df[correct_order]
        for i in C_dict:
            C_dict[i] = DataFrame(C_dict[i], columns=df.columns)[correct_order].values
        print("df and C are re-ordered")
    else:
        df0 = df.copy()

    # check4: Check rank condition of the constraint matrices and drop redundant constraints
    u = len(df.loc[:, df.isna().any()].columns)  # number of unknown variables
    for i in C_dict:
        Ci = C_dict[i]
        Ui = Ci.T[:u, :]  # unknown part of constraint Ci, want to drop column until full rank

        # if there are redundant columns, drop them
        Ui = array(Ui, dtype="float")
        if matrix_rank(Ui) < Ui.shape[1]:  # if columns are not full rank
            Ui_pd = DataFrame(Ui)  # set up a dataframe so that column index is fixed
            for col in Ui_pd.columns:  # loop to drop redundant columns
                if matrix_rank(Ui_pd.drop(col, axis=1)) == matrix_rank(Ui_pd):  # if redundant
                    Ui_pd = Ui_pd.drop(col, axis=1)  # drop column
            idx_to_keep = Ui_pd.columns
            C_dict[i] = Ci[idx_to_keep, :]
            d_dict[i] = array(d_dict[i]).reshape(-1, 1)[idx_to_keep, :]

        # if there are no free variables, forecast can be solved by the constraints without 1st step
        Ui_new = C_dict[i].T[:u, :]
        if (
            Ui_new.shape[0] == Ui_new.shape[1]
        ):  # if Ui is square, Ui is invertible from the previous step
            raise ValueError("Error: system exactly identified, no need to use ax")

    # check5: check C and d are consistent
    u = len(df.loc[:, df.isna().any()].columns)  # number of unknown variables
    for i in C_dict:
        Ci = C_dict[i]
        di = array(d_dict[i]).reshape(-1, 1)
    assert Ci.shape[0] == di.shape[0]

    # 1st step forecast
    df1, df0aug_fitted_model = unconstrained_forecast(df0, lag, Tin, model_list=estimators)

    # 2nd step reconciliation
    df2 = forecast_reconciliation(df1, df0, Tin, C_dict, d_dict)

    # put back the variables in the original order
    df1 = df1[df.columns]
    df2 = df2[df.columns]

    # keep the old output structure for consistency (to be removed later)
    return df2, df1, df0aug_fitted_model
