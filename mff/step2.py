import pandas as pd
import numpy as np
from scipy.linalg import block_diag
from sklearn.linear_model import LinearRegression
from itertools import combinations
from .utils import multiindex_from_multiindex_product

import warnings

from .covariance_calc import delete_exogenous_islands


def make_linear_extrapolation(series):
    # find missing years to fill in - should move out
    series.index = series.index.droplevel('subperiod')
    x_pred = series[series.isna()].index.values.reshape(-1, 1)

    # observed years
    d = series.dropna()
    x = d.index.values.reshape(-1, 1)
    y = d.values.reshape(-1, 1)

    # fit model
    model = LinearRegression()
    model.fit(x, y)

    # generate prediction
    y_hat = model.predict(x_pred)
    y_hat = pd.Series(y_hat.flatten(), index=x_pred.flatten(), name=d.name)
    return y_hat


def make_single_series_forecast(series):
    y_hat = make_linear_extrapolation(series)
    return y_hat


def make_forecasts(data, endog_list):
    endog_fcasts = [make_single_series_forecast(data[endog]) for endog in endog_list]
    endog_fcasts = pd.concat(endog_fcasts, axis=1)
    return endog_fcasts


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


class Reconciler:
    def __init__(self, data_all, exog, W, constraints, constants, lam):
        self.data = data_all.copy()
        self.exog = exog.copy()
        self.constraints = constraints
        self.constants = constants

        self.constraints.index.name = 'constraint'
        self.constants.index.name = 'constraint'

        self.W = W

        self.endog_idx = self.data.index.drop(self.exog.index)
        self.state_space_idx = self._make_state_space_idx()
        self.filled_endog_idx = self._make_filled_endog_idx()

        self.relative_freq = self.state_space_idx.to_frame(index=0)[['freq', 'subperiod']].drop_duplicates().groupby(
            'freq').count().to_dict()['subperiod']
        self.nvars = len(self.state_space_idx.to_frame(index=0)[['variable']].drop_duplicates())

        self.C_extended = constraints
        self.d_extended = constants
#        self.C_extended, self.d_extended = self._extend_constraints_matrix()
#        warn_lin_dep_rows(self.C_extended)

        assert isinstance(lam, (int, float, complex)) and not isinstance(lam, bool), 'lambda should be a numeric value'
        self._lam = lam

    def _make_filled_endog_idx(self):
        temp_grouper = self.endog_idx.to_frame(index=False).groupby(['variable', 'freq'])['year'].agg(['max', 'min'])
        temp_grouper_freq = self.state_space_idx.to_frame().reset_index(drop=True).groupby('freq')['subperiod'].max().to_dict()
        self.n_periods = temp_grouper['max'] - (temp_grouper['min'] - 1)
        idx = []
        for row in temp_grouper.iterrows():
            lower = row[1]['min']
            upper = row[1]['max']
            freq = row[0][1]  # pull out freq - order fixed from groupby above
            n_subperiods = temp_grouper_freq[freq]
            idx_temp = pd.DataFrame([row[0]] * (upper - lower + 1) * n_subperiods,
                                    index=pd.MultiIndex.from_product([range(lower, upper + 1),
                                                                      range(1, n_subperiods + 1)],
                                                                     names=['year', 'subperiod']
                                                                     ),
                                    columns=['variable', 'freq'])
            idx_temp = idx_temp.reset_index()
            idx.append(idx_temp)
        return pd.concat(idx).set_index(['year', 'subperiod', 'variable', 'freq']).index


    @property
    def lam(self):
        return self._lam

    @lam.setter
    def lam(self, value):
        self._lam = value

    def _make_state_space_idx(self):
        return self.data.index

    @staticmethod
    def collapse_multiindex(index, separator='_'):
        return [separator.join(map(str, idx)) for idx in index]

    def _extend_constraints_matrix(self):
        state_space_idx = self.state_space_idx

        C_dict = {}
        d_dict = {}
        constraints = self.constraints.copy()
        d = constraints['constant']
        C = constraints.drop('constant', axis=1, level='variable')

        for i in range(len(state_space_idx.names) - 1, 0,
                       -1):  # TODO: Append when no wildcards  # Backwards from most wildcards (if you start from narrow, you will get the multivariable wildcards too)
            for cols in combinations(state_space_idx.names, i):
                cols_to_extend = C.T.reset_index()[list(cols)].isna().all(axis=1).values

                C_subset = C.iloc[:, cols_to_extend].droplevel(cols, axis=1)
                C_subset = C_subset[(C_subset != 0).any(axis=1)]

                if len(C_subset) == 0:  # go next combination if there's no subset using combination
                    continue

                C = C.drop(C_subset.index)

                # Broadcast constraint coefficients over the wildcards
                drop_cols = state_space_idx.names.difference(cols)
                temp_idx = state_space_idx.droplevel(drop_cols).unique()
                C_np = np.kron(np.eye(len(temp_idx)), C_subset)

                # Convert numpy to dataframe
                constraints_col = multiindex_from_multiindex_product(temp_idx, C_subset.columns)
                constraints_idx = multiindex_from_multiindex_product(temp_idx, C_subset.index)
                C_extended = pd.DataFrame(C_np, index=constraints_idx,
                                          columns=constraints_col).reorder_levels(state_space_idx.names, axis=1)

                C_extended.index = self.collapse_multiindex(C_extended.index)
                name = ','.join(list(drop_cols))  # collapse for dict key
                C_dict[name] = C_extended

                # Broadcast constant over wildcards
                d_extended = pd.concat([d.loc[C_subset.index]] * len(temp_idx))
                d_extended.index = C_extended.index
                d_dict[name] = d_extended.iloc[:, 0]

                if len(C) == 0:  # stop if gone through all the constraints
                    break
            if len(C) == 0:
                break

        C_exog = pd.DataFrame((np.diag(self.exog) != 0).astype(float),
                              columns=self.exog.index,
                              index=self.exog.index
                              )  # TODO: standardize order of columns
        d_exog = self.exog
        C_exog.index = self.collapse_multiindex(C_exog.index)
        d_exog.index = self.collapse_multiindex(d_exog.index)

        endog_cols = state_space_idx.difference(C_exog.columns)
        C_exog[endog_cols] = 0.

        C_dict['exog'] = C_exog
        d_dict['exog'] = d_exog

        C = pd.concat(C_dict, axis=0).fillna(0)
        d = pd.concat(d_dict, axis=0).fillna(0)

        return C, d

    def make_F_single(self, nobs):
        return np.eye(nobs)

    def make_F_full(self, nobs):
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

    def make_F(self, nobs):
        if nobs >= 4:
            F = self.make_F_full(nobs)
        else:
            F = self.make_F_single(nobs)
        return F

    def make_phi(self, lam, F):
        lambdas = np.diag(self.nvars * [lam])
        phi = np.kron(lambdas, F)
        return phi

    def _fit(self):
        level_order = ['variable', 'freq', 'year', 'subperiod']
        constraints_idx = self.state_space_idx
        C = self.C_extended

        phis = []
        n_periods = self.n_periods.sort_index(level=['variable', 'freq'])
        for v in n_periods.values:
            F_temp = self.make_F(v)
            phi_temp = F_temp  # self.make_phi(self.lam ** v, F_temp)
            phis.append(phi_temp)

        sorted_idx = self.filled_endog_idx.to_frame().sort_index(level=['variable', 'freq', 'year', 'subperiod']).index
        phi = block_diag(*phis)
        phi = pd.DataFrame(phi, index=sorted_idx, columns=sorted_idx)

        W_inv = invert_df(self.W, matrix_name='W')
        denom = invert_df(W_inv + phi)

        fcasts_stacked = self.data.reorder_levels(denom.index.names)
        C = C.reorder_levels(denom.index.names, axis=1)
        cWc_inv = invert_df(C @ denom @ C.T)

        identity_df = pd.DataFrame(np.eye(len(denom)),
                                   columns=sorted_idx,
                                   index=sorted_idx)

        hp_component = (identity_df - (denom @ C.T @ cWc_inv @ C)) @ denom @ W_inv @ fcasts_stacked
        reconciliation_component = denom @ C.T @ cWc_inv @ self.d_extended

        y_adj = hp_component + reconciliation_component
        y_adj = y_adj.unstack('variable')

        return y_adj


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
    y_adj = reconciler._fit()

    err = np.abs(y_adj['y'] - y_adj.drop('y', axis=1).sum(axis=1)).mean()
    print(f'Avg reconcilation error for GDP accounting: {err}')

    y_agg = y_adj.groupby(['freq', 'year']).mean()
    err = (y_agg.loc['q'] - y_agg.loc['a']).mean().mean()
    print(f'Avg reconciliation error for quarters: {err}')
