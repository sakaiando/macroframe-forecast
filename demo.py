import pandas as pd
import sys
import numpy as np
sys.path.append(r'/mff')

from mff.mff.step2 import process_raw_constraints, Reconciler
from mff.mff.unconstrained_forecast import unconstrained_forecast
from mff.mff.covariance_calc import calculate_residuals, raw_covariance
from mff.mff.ecos_reader import load_excel

Tin = 10
forecast_year = 2022
lam = 100
n_horizons = 7

directory = r'./data/input.xlsx'
data, constraints_raw = load_excel(directory, excel_fmt='ecos')
#data = pd.read_excel(r'./data/input.xlsx', index_col=0, header=list(range(0, 3)), sheet_name='data').T
#constraints_raw = pd.read_excel(r'./data/input.xlsx', sheet_name='constraints', header=None, index_col=0)

#data.columns.name = 'variable'
#data = data.unstack(['freq', 'subperiod'])

C = process_raw_constraints(constraints_raw, index_iloc=range(0, 4))

cols = data.columns  # sktime can't handle MultiIndex columns, so store for later
data = data.T.reset_index(drop=True).T
data_nona = data.dropna()

y_hat, forecaster, fh = unconstrained_forecast(data_nona, Tin, fh=n_horizons, forecaster=None)
resids = calculate_residuals(data_nona, forecaster, n_horizons, cols)
W = raw_covariance(resids)


constraint_dict = {'c1': {'variables': ['BGS_BP6', 'BMS_BP6', 'constant'], 'constraint': [1, 1, 3]}}

constraint_1 = pd.DataFrame.from_records([(1, 1)], columns=pd.Index(['BGS_BP6', None]), index=['c1'])
y_hat.columns = cols
data.columns = cols

y_hat = y_hat.loc[forecast_year:]
y_exog = data.loc[y_hat.index]

y_hat = y_hat.stack(data.columns.names)
y_exog = y_exog.stack(data.columns.names)

reconciler = Reconciler(y_hat, y_exog, W, C, lam)
y_adj = reconciler._fit()

err = np.abs(y_adj['y'] - y_adj.drop('y', axis=1).sum(axis=1)).mean()
print(f'Avg reconcilation error for GDP accounting: {err}')

y_agg = y_adj.groupby(['freq', 'year']).mean()
err = (y_agg.loc['q'] - y_agg.loc['a']).mean().mean()
print(f'Avg reconciliation error for quarters: {err}')
