from numpy import array, concatenate, isnan
from numpy.linalg import inv
from pandas import DataFrame
from sklearn.covariance import OAS


def reconcile(df1: DataFrame, df0: DataFrame, Tin: int, C_dict: dict, d_dict: dict) -> DataFrame:
    """
     Parameters
     ----------
     df0: dataframe
         (T+h) x m dataframe representing input data
         the first m-k columns of T:T+h rows are nan, and the rest are not nan
     df1: dataframe
         (T+h) x m dataframe representing 1st stage forecast
         no columns contain nan
     Tin: int
         the number of time periods in historical data used to estimate forecast-error
     C_dict: dictionary
         T+h length dictionary, each element of which is
         (m-n) x m numpy matrix C in the constraint C x df = d
         the order of columns must be the same of the columns of df
         n is the number of free variables (net contents)
     d_dict: dictionary
         T+h length dictionary of numpy array, each element of which is
         (m-n) x 1 column vector in the constraint C x df = d

    Returns
     -------
     df2: dataframe
         (T+h) x m dataframe
         the last h rows are forecast that satisfies C x df=d
    """
    # extract information on T,h,u, from df0
    T = sum(~isnan(df0).any(axis=1))
    h = len(df0) - T
    df0_u = df0.loc[:, df0.isna().any()]
    u = len(df0_u.columns)
    df0_u = df0_u.to_numpy()

    df1_u = df1.iloc[:, :u].to_numpy()
    df1_k = df1.iloc[:, u:].to_numpy()

    # construct weight matrix
    eh = df1_u[T - Tin : T, :] - df0_u[T - Tin : T, :]  # in-sample one-step ahead forecast error
    W = OAS().fit(eh).covariance_

    # reconcili rh by projecting it on constraint
    df2_u = df1_u.copy()
    for hi in range(h):
        C = array(C_dict[T + hi], dtype="float")  # to avoid error in inv(U.T @ W @ U)
        U = C.T[:u, :]
        d = d_dict[T + hi]
        # this step may need python 3.8 or above, 3.6 may gives an error
        df2_u[T + hi : T + hi + 1, :] = (
            df1_u[T + hi : T + hi + 1, :].T
            - W
            @ U
            @ inv(U.T @ W @ U)
            @ (
                C
                @ concatenate(
                    (df1_u[T + hi : T + hi + 1, :], df1_k[T + hi : T + hi + 1, :]), axis=1
                ).T
                - d
            )
        ).T

    df2 = concatenate([df2_u, df1_k], axis=1)
    df2 = DataFrame(df2, index=df1.index, columns=df1.columns)
    df2.iloc[:T, :] = df0.iloc[:T, :]

    return df2
