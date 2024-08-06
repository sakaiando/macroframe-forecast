import pandas as pd
import sys
import shutil
import numpy as np


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

    resids = calculate_oos_residuals(data, forecaster, n_horizons, cols_dict)
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


if __name__ == '__main__':
    import xlwings as xw
    import os

    os.chdir(r'C:\Users\dbilgin\OneDrive - International Monetary Fund (PRD)\prototype\mff')
    Tin = 5

    lam = 1e7
    directory = r'./data/input.xlsm'

    with xw.App(visible=True, add_book=False) as app:
        print('Reading data')
        wb = xw.Book(directory).set_mock_caller()

        data, constraints_raw = load_excel(directory, data_fmt='ecos', constraint_fmt='readable')
        print('Read data')
        state_space, conditional_constraints, forecast_start, n_horizons = calculate_initial_inputs(data)
        constraints = conditional_constraints + constraints_raw
        C, b = generate_constraint_mat_from_equations(constraints, state_space)

        print('Constraints compiled')
        y_hat, W = step1_with_multiindex_col_data(data, n_horizons)

        y_hat, y_exog = prep_fcast_and_exog(y_hat, data, forecast_start)
        print('Forecasts generated')

        #y_hat.update(y_exog)
        reconciler = Reconciler(y_hat, y_exog, W, C, b, lam)
        y_adj = reconciler._fit()

        data.update(y_adj.unstack(['freq', 'subperiod']), overwrite=False)
        df_out = data.reset_index().T.reset_index('variable').values.tolist()

        write_to_excel(df_out)

    # err = np.abs(y_adj['y'] - y_adj.drop('y', axis=1).sum(axis=1)).mean()
    # print(f'Avg reconcilation error for GDP accounting: {err}')
    #
    # y_agg = y_adj.groupby(['freq', 'year']).mean()
    # err = (y_agg.loc['q'] - y_agg.loc['a']).mean().mean()
    # print(f'Avg reconciliation error for quarters: {err}')
