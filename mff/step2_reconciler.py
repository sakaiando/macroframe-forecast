import pandas as pd
import numpy as np
from scipy.linalg import block_diag
from sklearn.linear_model import LinearRegression

from itertools import product, combinations
from .step0_parse_constraints import find_first_na_in_df

import warnings


class Reconciler:
    def __init__(self, fcast_all, exog, W, constraints, constants, lam, n_hist_points=2):
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

        self.C_extended = constraints
        self.d_extended = constants

        assert isinstance(lam, (int, float, complex)) and not isinstance(lam, bool), 'lambda should be a numeric value'
        self._lam = lam
        self.y_reconciled = None

    @property
    def lam(self):
        return self._lam

    @lam.setter
    def lam(self, value):
        self._lam = value

    @staticmethod
    def collapse_multiindex(index, separator='_'):
        return [separator.join(map(str, idx)) for idx in index]

    @staticmethod
    def multiindex_from_multiindex_product(idx_left, idx_right):
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
        lambdas = np.diag(self.nvars * [lam])
        phi = np.kron(lambdas, F)
        return phi

    def W_extended(self):
        idx_to_add = self.state_space_idx.drop(self.W.index)
        W = self.W.copy()
        W[idx_to_add] = 0
        W = W.T
        W[idx_to_add] = 0
        return W

    def fit(self):
        level_order = ['freq', 'variable', 'year', 'subperiod']
        constraints_idx = self.state_space_idx
        C = self.C_extended

        phis = []
        for k, v in self.relative_freq.items():
            F_temp = self.make_F(v * (self.nperiods))
            phi_temp = self.make_phi(self.lam ** v, F_temp)
            phis.append(phi_temp)

        phi_rescaler = pd.Series(1/np.diag(self.W_extended().replace({0: 1})), index=self.W_extended().index).groupby(['variable', 'freq', 'subperiod']).min()

        sorted_idx = constraints_idx.sortlevel(level_order)[0]
        phi = block_diag(*phis)
        phi = pd.DataFrame(phi, index=sorted_idx, columns=sorted_idx)

        phi = phi.multiply(phi_rescaler)

        W_inv = invert_df(self.W_extended())
        denom = invert_df(W_inv + phi)

        fcasts_stacked = self.data.reorder_levels(denom.index.names)
        C = C.reorder_levels(denom.index.names, axis=1)
        cWc_inv = invert_df(C @ denom @ C.T)

        identity_df = pd.DataFrame(np.eye(len(denom)),
                                   columns=sorted_idx,
                                   index=sorted_idx)

        y_adj = (identity_df - (denom @ C.T @ cWc_inv @ C)) @ denom @ W_inv @ fcasts_stacked + denom @ C.T @ cWc_inv @ self.d_extended

        # note that y_adj may not contain correct values for known variables due to indeterminacy. this does not affect forecasts for unknown variables
        y_adj = y_adj.unstack('variable')
        y_adj = y_adj.unstack(['freq', 'subperiod'])
        y_adj.update(self.exog.unstack('year').T)
        self.y_reconciled = y_adj

def warn_lin_dep_rows(df, tol=1e-8):
    u, eigval, v = np.linalg.svd(df)
    for i in np.where(eigval < tol)[0]:
        lin_dep_idx = np.where(np.abs(v[i]) > tol)[0]
        warnings.warn(f'The rows {df.iloc[lin_dep_idx].index.to_list()} are linearly dependent')


def invert_df(df, matrix_name=None):
    if matrix_name is not None:
        warn_lin_dep_rows(df)
    return pd.DataFrame(np.linalg.pinv(df), columns=df.columns, index=df.index)


def process_raw_constraints(constraints_raw, index_iloc=range(0, 3)):
    constraints = constraints_raw.T.set_index(constraints_raw.index[
                                                  index_iloc].to_list()).T  # pick levels for columns (can't do in read_excel because it propegates the values where subsequent columns are missing)
    constraints = constraints.fillna(0)

    return constraints


if __name__ == '__main__':
    data = pd.read_excel(r'./data/input.xlsx', index_col=0, header=list(range(0, 3)), sheet_name='data').T
    data.columns.name = 'variable'
    data = data.stack()

    constraints_raw = pd.read_excel(r'./data/input.xlsx', sheet_name='constraints', header=None, index_col=0)
    forecast_start = 2023
    lam = 100

    C = process_raw_constraints(constraints_raw, index_iloc=range(0, 4))

    full_data = data.unstack().stack(dropna=False).fillna(0)
    exog_data = data
    W = pd.DataFrame(np.eye(len(full_data)), index=full_data.index, columns=full_data.index)
    reconciler = Reconciler(full_data, exog_data, W, C, lam)
    y_adj = reconciler.fit()

    err = np.abs(y_adj['y'] - y_adj.drop('y', axis=1).sum(axis=1)).mean()
    print(f'Avg reconcilation error for GDP accounting: {err}')

    y_agg = y_adj.groupby(['freq', 'year']).mean()
    err = (y_agg.loc['q'] - y_agg.loc['a']).mean().mean()
    print(f'Avg reconciliation error for quarters: {err}')
