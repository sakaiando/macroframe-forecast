import pandas as pd
import numpy as np
from scipy.linalg import block_diag


from itertools import product
from .step0_parse_constraints import find_first_na_in_df

import warnings


class Reconciler:
    """
        A class to reconcile forecast data with exogenous variables and constraints based on Ando (2024)

        References:
            Ando, Sakai (2024), “Smooth Forecast Reconciliation,” IMF Working Paper.

        Attributes:
            start_date (datetime): The start date for the reconciliation dataframe.
            data (pd.Series): A vector containing all forecasted values (should be the full state-space of forecasts).
            exog (pd.Series): A vector containing individual forecast values that should be constrained.
            constraints (pd.DataFrame): Matrix containing constraint coefficents.
            constants (pd.DataFrame): Vector containing constants for constraints.
            W (pd.DataFrame): Covariance matrix of unknown variable forecast errors. Dimensions should be the full forecast horizon for any variable that is unknown at any point in the forecast horizon.
            state_space_idx (pd.MultiIndex): pd.MultiIndex containing the state space with columns ['variable', 'year', 'freq', 'subperiod'].
            nperiods (int): The number of forecast periods.
            relative_freq (dict): The relative frequency of subperiods.
            nvars (int): The number of distinct variables.
            _lam (float): Smoothing parameter based on HP filter smoothing. Default is 100.
            y_reconciled (pd.DataFrame): Datafrane of reconciled forecasts.
        """

    def __init__(self, fcast_all, exog, W, constraints, constants, lam, n_hist_points=2):
        """
        Initializes the Reconciler class with forecast data, exogenous variables, constraints, and other parameters. Infers the number of variables, forecast horizons and relative frequencies of 'freq'

        Args:
           fcast_all (pd.DataFrame): Dataframe containing forecasted values. Must not have NAs and contain columns/indexes with names ['variable', 'year', 'freq', 'subperiod'].
           exog (pd.DataFrame): Dataframe containing individual values to fix.
           W (pd.DataFrame): Covariance matrix of unknown variable forecast errors. Dimensions should be the full forecast horizon for any variable that is unknown at any point in the forecast horizon.
           constraints (pd.DataFrame): The constraints for the reconciliation.
           constants (pd.DataFrame): The constants for the reconciliation.
           lam (float): The lambda value for the reconciliation.
           n_hist_points (int, optional): The number of historical points to consider. Defaults to 2.
        """
        self.start_date = find_first_na_in_df(exog) - n_hist_points

        self.data = fcast_all.copy().loc[self.start_date:].T.stack()
        self.exog = exog.copy().loc[self.start_date:].T.stack()
        self.constraints = constraints
        self.constants = constants

        self.constraints.index.name = 'constraint'
        self.constants.index.name = 'constraint'

        self.W = W

        self.state_space_idx = self.data.index
        self.nperiods = len(self.state_space_idx.to_frame(index=False)['year'].drop_duplicates())
        self.relative_freq = self.state_space_idx.to_frame(index=0)[['freq', 'subperiod']].drop_duplicates().groupby(
            'freq').count().to_dict()['subperiod']
        self.nvars = len(self.state_space_idx.to_frame(index=0)[['variable']].drop_duplicates())

        assert isinstance(lam, (int, float, complex)) and not isinstance(lam, bool), 'lambda should be a numeric value'
        self._lam = lam
        self.y_reconciled = None

    @property
    def lam(self):
        """
        Gets the lambda value.

        Returns:
            float: The lambda value.
        """
        return self._lam

    @lam.setter
    def lam(self, value):
        """
        Sets the lambda value.

        Returns:
            float: The lambda value.
        """
        self._lam = value

    @staticmethod
    def collapse_multiindex(index, separator='_'):
        """
        Collapses a MultiIndex into a single index with a specified separator.

        Args:
           index (pd.MultiIndex): The MultiIndex to collapse.
           separator (str, optional): The separator to use. Defaults to '_'.

        Returns:
           list: The collapsed index.
        """
        return [separator.join(map(str, idx)) for idx in index]

    @staticmethod
    def multiindex_from_multiindex_product(idx_left, idx_right):
        """
        Creates a MultiIndex from the product of two MultiIndexes.

        Args:
           idx_left (pd.MultiIndex): The left MultiIndex.
           idx_right (pd.MultiIndex): The right MultiIndex.

        Returns:
           pd.MultiIndex: The resulting MultiIndex.
        """
        concat_list = []
        left_is_multiindex = isinstance(idx_left, pd.MultiIndex)
        right_is_multiindex = isinstance(idx_right, pd.MultiIndex)
        for a, b in product(idx_left, idx_right):
            if left_is_multiindex and right_is_multiindex:
                item = (*a, *b)
            elif left_is_multiindex and not right_is_multiindex:
                item = (*a, b)
            elif not left_is_multiindex and right_is_multiindex:
                item = (a, *b)
            else:
                item = (a, b)
            concat_list.append(item)
        return pd.MultiIndex.from_tuples(concat_list, names=list(idx_left.names) + list(idx_right.names))

    def make_F(self, nobs):
        """
        Creates the HP filter smoothing matrix for the reconciliation process. (Equation 4, Ando 2023)

        Args:
            nobs (int): The number of observations.

        Returns:
            np.ndarray: The F matrix.
        """
        # make (off) diagonal elements
        F = np.zeros((nobs, nobs))
        for i, v in enumerate([1., -4, 6., -4., 1.]):
            F = F + v * np.eye(nobs, k=i - 2)

        # replace first and last rows
        row1 = np.array([1, -2, 1])
        row2 = np.array([-2, 5, -4, 1])

        F[0, :3] = row1
        F[-1, -3:] = row1[::-1]  # reverse array order
        F[1, :4] = row2
        F[-2, -4:] = row2[::-1]
        return F

    def make_phi(self, lam, F):
        """
        Creates the phi matrix for the reconciliation process. (Equation 8, Ando 2024)

        Args:
            lam (float): The lambda value.
            F (np.ndarray): The F matrix.

        Returns:
            np.ndarray: The phi matrix.
        """
        lambdas = np.diag(self.nvars * [lam])
        phi = np.kron(lambdas, F)
        return phi

    def W_extended(self):
        """
        Extends the weight matrix W to match the state space index. Fills in missing values with identity matrix.

        Returns:
            pd.DataFrame: The extended weight matrix.
        """
        idx_to_add = self.state_space_idx.drop(self.W.index)
        W = self.W.copy()
        W[idx_to_add] = 0
        W = W.T
        W[idx_to_add] = 0
        W.loc[idx_to_add, idx_to_add] = np.eye(len(idx_to_add))
        return W

    def fit(self):
        """
        Adjusts the unreconciled forecasts conform to forecast constraints based on Equation 11 in Ando (2024).
        """
        level_order = ['freq', 'variable', 'year', 'subperiod']
        constraints_idx = self.state_space_idx
        C = self.constraints


        # create Phi matrix for HP smoother
        phis = []
        for k, v in self.relative_freq.items():
            F_temp = self.make_F(v * (self.nperiods))
            phi_temp = self.make_phi(self.lam ** v, F_temp)
            phis.append(phi_temp)

        sorted_idx = constraints_idx.sortlevel(level_order)[0]
        phi = block_diag(*phis)
        phi = pd.DataFrame(phi, index=sorted_idx, columns=sorted_idx)

        # Rescale HP filter by minimum variance for each variable
        phi_rescaler = pd.Series(1/np.diag(self.W_extended().replace({0: 1})), index=self.W_extended().index).groupby(['variable', 'freq', 'subperiod']).min()
        phi = phi.multiply(phi_rescaler)


        # precompute parts of equation (11)
        W_inv = invert_df(self.W_extended())
        denom = invert_df(W_inv + phi)

        fcasts_stacked = self.data.reorder_levels(denom.index.names)
        C = C.reorder_levels(denom.index.names, axis=1)
        cWc_inv = invert_df(C @ denom @ C.T)

        identity_df = pd.DataFrame(np.eye(len(denom)),
                                   columns=sorted_idx,
                                   index=sorted_idx)

        y_adj = (identity_df - (denom @ C.T @ cWc_inv @ C)) @ denom @ W_inv @ fcasts_stacked + denom @ C.T @ cWc_inv @ self.constants  # equation (11)

        # note that y_adj may not contain correct values for known variables due to indeterminacy. this does not affect forecasts for unknown variables
        y_adj = y_adj.unstack('variable')
        y_adj = y_adj.unstack(['freq', 'subperiod'])
        y_adj.update(self.exog.unstack('year').T)
        self.y_reconciled = y_adj

def warn_lin_dep_rows(df, tol=1e-8):
    """
        Warns about linearly dependent rows in a DataFrame. Linear dependence is determined using Singular Value
        Decomposition, returning eigenvectors corresponding to 0 eigenvalues.

        Args:
            df (pd.DataFrame): The DataFrame to check.
            tol (float, optional): The tolerance for detecting linear dependence. Defaults to 1e-8.
        """
    u, eigval, v = np.linalg.svd(df)
    for i in np.where(eigval < tol)[0]:
        lin_dep_idx = np.where(np.abs(v[i]) > tol)[0]
        warnings.warn(f'The rows {df.iloc[lin_dep_idx].index.to_list()} are linearly dependent')


def invert_df(df, matrix_name=None):
    """
        Inverts a DataFrame using the pseudo-inverse. Will warn about linearly independent rows if matrix_name is not
        None.

        Args:
            df (pd.DataFrame): The DataFrame to invert.
            matrix_name (str, optional): The name of the matrix for warning purposes. Defaults to None.

        Returns:
            pd.DataFrame: The inverted DataFrame.
        """
    if matrix_name is not None:
        warn_lin_dep_rows(df)
    return pd.DataFrame(np.linalg.pinv(df), columns=df.columns, index=df.index)


if __name__ == '__main__':
    data = pd.read_excel(r'./data/input.xlsx', index_col=0, header=list(range(0, 3)), sheet_name='data').T
    data.columns.name = 'variable'
    data = data.stack()

    constraints_raw = pd.read_excel(r'./data/input.xlsx', sheet_name='constraints', header=None, index_col=0)
    forecast_start = 2023
    lam = 100

    # TODO: Unit test still to be implemented
