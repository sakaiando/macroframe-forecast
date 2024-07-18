import pandas as pd
import sys
import numpy as np
sys.path.append(r'/mff')

from mff.mff.step2 import Reconciler
from mff.mff.unconstrained_forecast import unconstrained_forecast
from mff.mff.covariance_calc import calculate_residuals, raw_covariance
from mff.mff.ecos_reader import load_excel
from mff.mff.string_parser import generate_constraints_from_equations

def calculate_state_space(ss_idx):
    ss = ss_idx.to_frame().astype(str)
    ss = ss['variable'] + '_' + ss['year'] + (ss['freq'] + ss['subperiod']).replace({'A1': ''})
    return ss.tolist()


def calculate_forecast_interval(ss):
    return ss.get_level_values('year').min(), ss.get_level_values('year').max()


def calculate_initial_inputs(df):
    df_stacked = df.stack(df.columns.names, dropna=False)
    ss_idx = df_stacked.index[df_stacked.isna()]

    ss_str = calculate_state_space(ss_idx)
    forecast_start, forecast_end = calculate_forecast_interval(ss_idx)
    n_horizons = int(forecast_end - forecast_start) + 1
    return ss_str, forecast_start, n_horizons


if __name__ == '__main__':
    import xlwings as xw
    import os
    os.chdir(r'C:\Users\dbilgin\OneDrive - International Monetary Fund (PRD)\prototype')

    Tin = 10
    lam = 100

    directory = r'./data/input.xlsx'
    xw.Book(directory).set_mock_caller()

    data, constraints_raw = load_excel(directory, data_fmt='ecos', constraint_fmt='readable')
    state_space, forecast_start, n_horizons = calculate_initial_inputs(data)
    C, b = generate_constraints_from_equations(constraints_raw, state_space)

    cols = data.columns  # sktime can't handle MultiIndex columns, so store for later
    data = data.T.reset_index(drop=True).T
    data_nona = data.dropna()

    y_hat, forecaster, fh = unconstrained_forecast(data_nona, Tin, fh=n_horizons, forecaster=None)
    resids = calculate_residuals(data_nona, forecaster, n_horizons, cols)
    W = raw_covariance(resids)

    y_hat.columns = cols
    data.columns = cols

    y_hat = y_hat.loc[forecast_start:]
    y_exog = data.loc[y_hat.index]

    y_hat = y_hat.stack(data.columns.names)
    y_exog = y_exog.stack(data.columns.names)

    reconciler = Reconciler(y_hat, y_exog, W, C, b, lam)
    y_adj = reconciler._fit()

    data.columns = cols
    data.update(y_adj.unstack(['freq', 'subperiod']))
    wb = xw.Book().caller()
    data_out = data.reset_index().T.reset_index('variable')
    sheet = wb.sheets['data_out']
    sheet.range('A1').value = data_out.values

    err = np.abs(y_adj['y'] - y_adj.drop('y', axis=1).sum(axis=1)).mean()
    print(f'Avg reconcilation error for GDP accounting: {err}')

    y_agg = y_adj.groupby(['freq', 'year']).mean()
    err = (y_agg.loc['q'] - y_agg.loc['a']).mean().mean()
    print(f'Avg reconciliation error for quarters: {err}')
