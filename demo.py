import pandas as pd
import sys
import numpy as np
sys.path.append(r'/mff')

from mff.mff.step2 import Reconciler
from mff.mff.unconstrained_forecast import unconstrained_forecast
from mff.mff.covariance_calc import calculate_residuals, raw_covariance
from mff.mff.ecos_reader import load_excel
from mff.mff.string_parser import generate_constraints_from_equations
from itertools import product

Tin = 10
forecast_year = 2023
lam = 100
n_horizons = 7

directory = r'./data/input.xlsx'
data, constraints_raw, state_space = load_excel(directory, data_fmt='ecos', constraints_fmt='readable')
C, b = generate_constraints_from_equations(constraints_raw.values, state_space)

cols = data.columns  # sktime can't handle MultiIndex columns, so store for later
data = data.T.reset_index(drop=True).T
data_nona = data.dropna()

y_hat, forecaster, fh = unconstrained_forecast(data_nona, Tin, fh=n_horizons, forecaster=None)
resids = calculate_residuals(data_nona, forecaster, n_horizons, cols)
W = raw_covariance(resids)

y_hat.columns = cols
data.columns = cols

y_hat = y_hat.loc[forecast_year:]
y_exog = data.loc[y_hat.index]

y_hat = y_hat.stack(data.columns.names)
y_exog = y_exog.stack(data.columns.names)

reconciler = Reconciler(y_hat, y_exog, W, C, lam, b)
y_adj = reconciler._fit()

err = np.abs(y_adj['y'] - y_adj.drop('y', axis=1).sum(axis=1)).mean()
print(f'Avg reconcilation error for GDP accounting: {err}')

y_agg = y_adj.groupby(['freq', 'year']).mean()
err = (y_agg.loc['q'] - y_agg.loc['a']).mean().mean()
print(f'Avg reconciliation error for quarters: {err}')
