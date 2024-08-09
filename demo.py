import numpy as np
import pandas as pd
import sys
import shutil

sys.path.append(r'')

from mff.step2 import Reconciler
from mff.unconstrained_forecast import staggered_forecast
from mff.covariance_calc import calc_covariance, calculate_oos_residuals
from mff.ecos_reader import load_excel
from mff.string_parser import generate_constraint_mat_from_equations
from sktime.utils import mlflow_sktime


def calculate_state_space(ss_idx):
    ss = ss_idx.to_frame().astype(str)
    ss = ss['variable'] + '_' + ss['year'] + (ss['freq'] + ss['subperiod']).replace({'A1': ''})
    return ss.tolist()


def calculate_forecast_interval(ss):
    return ss.get_level_values('year').min(), ss.get_level_values('year').max()


def calculate_initial_inputs(df, n_start_years=1):
    na_years = df.index[df.isna().any(axis=1)]
    forecast_start, forecast_end = calculate_forecast_interval(na_years)
    n_horizons = int(forecast_end - forecast_start)

    df_stacked = df.stack(df.columns.names, dropna=False)
    fcast_stacked = df_stacked[df_stacked.index.get_level_values('year') >= forecast_start - n_start_years]
    ss_str = calculate_state_space(fcast_stacked.index)

    fixed_fcasts = fcast_stacked.dropna()
    conditional_fcast_constraints = [f'{i} - {j}' for i, j in
                                     zip(calculate_state_space(fixed_fcasts.index), fixed_fcasts.values)]
    return ss_str, conditional_fcast_constraints, forecast_start, n_horizons


def write_to_excel(values_out, out_sheet='data'):
    wb = xw.Book().caller()
    sheet = wb.sheets[out_sheet]

    num_rows = len(values_out)
    num_cols = len(values_out[0]) if values_out else 0

    # Iterate over the values and write them with formatting if necessary
    for i in range(num_rows):
        for j in range(num_cols):
            cell = sheet.range((i + 1, j + 1))  # (i + 1, j + 1) is used because Excel is 1-indexed
            if cell.value is None:  # Check if the cell is empty
                cell.api.Font.Color = 255

    sheet.range('A1').value = values_out


def make_df_checksum(df):
    return pd.util.hash_pandas_object(df).sum()


def create_cache(forecaster, df, dir=r'cache'):
    shutil.rmtree(dir)
    os.mkdir(dir)
    mlflow_sktime.save_model(forecaster, os.path.join(dir, 'model.pickle'))
    checksum = make_df_checksum(df)
    with open(os.path.join(dir, 'df_checksum.txt'), 'w') as f:
        f.write(str(checksum))


def check_df_unchanged(df, checksum_dir=r'.\cache\df_checksum.txt'):
    checksum = make_df_checksum(df)
    with open(checksum_dir) as f:
        checkcurrent = f.read()
    return checksum == int(checkcurrent)


def step1_with_multiindex_col_data(data, n_horizons):
    cols = data.columns  # sktime can't handle MultiIndex columns, so store for later
    cols_dict = {i: col for i, col in enumerate(cols)}
    data = data.T.reset_index(drop=True).T

    y_hat, forecaster, fh = staggered_forecast(data, forecaster=None)

    resids = calculate_oos_residuals(data, forecaster, n_horizons, cols_dict, n_periods=4)
    W = calc_covariance(resids)

    y_hat.columns = cols
    data.columns = cols
    return y_hat, W


def prep_fcast_and_exog(y_hat, data, forecast_start, n_start_years=1):
    y_hat = y_hat.loc[forecast_start - n_start_years:]
    y_exog = data.loc[y_hat.index]

    y_hat = y_hat.stack(data.columns.names)
    y_exog = y_exog.stack(data.columns.names)
    return y_hat, y_exog


def run(Tin, lam):
    import xlwings as xw
    import os

    os.chdir(r'C:\Users\dbilgin\OneDrive - International Monetary Fund (PRD)\prototype\mff')

    directory = r'./data/input.xlsm'

    print('Reading data')
    with xw.App(visible=False) as app:
        wb = xw.Book.caller()  # xw.Book(directory).set_mock_caller() for calling from script
        data, constraints_raw = load_excel(directory, data_fmt='ecos', constraint_fmt='readable')
        print('Read data')
        state_space, conditional_constraints, forecast_start, n_horizons = calculate_initial_inputs(data)
        constraints = conditional_constraints + constraints_raw
        C, b = generate_constraint_mat_from_equations(constraints, state_space)

        print('Constraints compiled')
        y_hat, W = step1_with_multiindex_col_data(data, n_horizons)

        y_hat, y_exog = prep_fcast_and_exog(y_hat, data, forecast_start)
        print('Forecasts generated')

        reconciler = Reconciler(y_hat, y_exog, W, C, b, lam)
        y_adj = reconciler._fit()

        data.update(y_adj.unstack(['freq', 'subperiod']), overwrite=False)
        df_out = data.reset_index().T.reset_index('variable').values.tolist()

        write_to_excel(df_out)


def close_extra_workbooks(wb_not_close):
    # Get a reference to the current application
    app = xw.apps.active

    # Loop through all open workbooks
    for wb in app.books:
        # Define your criteria for closing workbooks
        # For example, close any workbook that is not named 'MainWorkbook.xlsx'
        if wb.name != wb_not_close:
            wb.close()


def load_synthetic_data():
    d = np.array([[4.19, 4.51, -0.32, 0.94, 6.21],
                  [2.69, 2.38, 0.3, 1.01, 7.87],
                  [-9.67, -10.22, 0.55, 0.98, 6.94],
                  [-8.8, -8.99, 0.19, 0.9, 4.44],
                  [-9.57, -10.26, 0.69, 0.86, 4.08],
                  [-4.47, -4.12, -0.34, 0.73, -0.11],
                  [-3.08, -2.27, -0.81, 0.65, 1.17],
                  [-7.98, -7.65, -0.33, 0.62, 3.25],
                  [-7.6, -6.86, -0.74, 0.67, 4.51],
                  [-4.89, -1.53, -3.36, 0.82, 5.5],
                  [-5.85, -2.31, -3.54, 0.94, 5.28],
                  [-7.29, -4.11, -3.17, 0.97, 6.62],
                  [-7.03, -3.45, -3.57, 1.02, 8.49],
                  [-4.8, -0.61, -4.19, 1.22, 10.83],
                  [-6.36, -2.31, -4.05, 1.42, 5.57],
                  [-3.44, -1.05, -2.4, 1.39, -5.46],
                  [-4.63, -1.06, -3.57, 1.33, 6.72],
                  [-4.87, -0.43, -4.45, 1.39, 2.67],
                  [0.93, 3.97, -3.05, 1.29, 1.32],
                  [1.85, 4.55, -2.7, 1.33, 0.63],
                  [1.14, 3.84, -2.7, 1.33, 2.7],
                  [-2.08, 1.16, -3.24, 1.11, 5.17],
                  [-2.73, 2.01, -4.75, 1.11, 1.94],
                  [-1.91, 1.75, -3.66, 1.13, 2.94],
                  [-2.2, 0.77, -2.96, 1.18, 4.03],
                  [-3.35, 0.09, -3.44, 1.12, 2.51],
                  [0.55, 2.11, -1.55, 1.14, -3.34],
                  [-2.46, 0.03, -2.49, 1.18, 4.79],
                  [-2.73, -0.27, np.nan, 1.05, 4.51],
                  [-2.98, -0.55, np.nan, 1.1, 4.26],
                  [np.nan, np.nan, np.nan, 1.1, 4.04],
                  [np.nan, np.nan, np.nan, 1.1, 3.86],
                  [np.nan, np.nan, np.nan, 1.1, 3.72],
                  [np.nan, np.nan, np.nan, 1.1, 3.60],
                  [np.nan, np.nan, np.nan, 1.1, 3.49],
                  [np.nan, np.nan, np.nan, 1.1, 3.39],
                  [np.nan, np.nan, np.nan, 1.1, 3.30]])
    idx = pd.Int64Index([1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004,
                         2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
                         2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026,
                         2027, 2028, 2029, 2030],
                        dtype='int64', name='year')
    col = pd.MultiIndex.from_tuples([('CA', 'A', 1),
                                     ('TA', 'A', 1),
                                     ('IA', 'A', 1),
                                     ('FX', 'A', 1),
                                     ('GDP', 'A', 1)],
                                    names=['variable', 'freq', 'subperiod'])
    data = pd.DataFrame(d, index=idx, columns=col)
    constraints = ['CA? - TA? - IA?']
    return data, constraints


if __name__ == '__main__':
    import xlwings as xw
    import os

    os.chdir(r'C:\Users\dbilgin\OneDrive - International Monetary Fund (PRD)\prototype\mff')
    Tin = 1
    lam = 1e2

    use_excel = False
    directory = r'./data/input.xlsm'

    print('Reading data')
    if use_excel:
        wb = xw.Book(directory).set_mock_caller()
        data, constraints_raw = load_excel(directory, data_fmt='ecos', constraint_fmt='readable')
    else:
        data, constraints_raw = load_synthetic_data()
    print('Read data')

    state_space, conditional_constraints, forecast_start, n_horizons = calculate_initial_inputs(data)
    constraints = conditional_constraints + constraints_raw
    C, b = generate_constraint_mat_from_equations(constraints, state_space)

    print('Constraints compiled')
    y_hat, W = step1_with_multiindex_col_data(data, n_horizons)

    y_hat, y_exog = prep_fcast_and_exog(y_hat, data, forecast_start)
    print('Forecasts generated')

    # y_hat.update(y_exog)
    reconciler = Reconciler(y_hat, y_exog, W, C, b, lam)
    y_adj = reconciler._fit()

    data.update(y_adj.unstack(['freq', 'subperiod']), overwrite=False)
    df_out = data.reset_index().T.reset_index('variable').values.tolist()

    if use_excel:
        write_to_excel(df_out)

    # err = np.abs(y_adj['y'] - y_adj.drop('y', axis=1).sum(axis=1)).mean()
    # print(f'Avg reconcilation error for GDP accounting: {err}')
    #
    # y_agg = y_adj.groupby(['freq', 'year']).mean()
    # err = (y_agg.loc['q'] - y_agg.loc['a']).mean().mean()
    # print(f'Avg reconciliation error for quarters: {err}')
