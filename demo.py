import pandas as pd
import sys
import shutil
import xlwings as xw

sys.path.append(r'')

from mff.step2_reconciler import Reconciler
from mff.step1_forecast import UnconditionalForecaster
from mff.ecos_reader import load_excel
from mff.string_parser import generate_constraint_mat_from_equations
from sktime.utils import mlflow_sktime
from mff.default_forecaster import get_default_forecaster


def calculate_state_space(ss_idx):
    ss = ss_idx.to_frame().astype(str)
    ss = ss['variable'] + '_' + ss['year'] + (ss['freq'] + ss['subperiod']).replace({'A1': ''})
    return ss.tolist()


def convert_exog_to_constraint(df, forecast_start=None):
    if forecast_start is None:
        forecast_start = df.index.min()
    df_stacked = df.stack(df.columns.names, dropna=False)
    fcast_stacked = df_stacked[df_stacked.index.get_level_values('year') >= forecast_start]
    ss_str = calculate_state_space(fcast_stacked.index)

    fixed_fcasts = fcast_stacked.dropna()
    conditional_fcast_constraints = [f'{i} - {j}' for i, j in
                                     zip(calculate_state_space(fixed_fcasts.index), fixed_fcasts.values)]
    return ss_str, conditional_fcast_constraints


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


def prep_fcast_and_exog(y_hat, data, forecast_start, n_start_years=1):
    y_hat = y_hat.loc[forecast_start - n_start_years:]
    y_exog = data.loc[y_hat.index]

    y_hat = y_hat.stack(data.columns.names)
    y_exog = y_exog.stack(data.columns.names)
    return y_hat, y_exog


def run(lam, n_resid=5, n_lags=4):
    print('Reading data')
    with xw.App(visible=False) as app:
        wb = xw.Book.caller()  # xw.Book(directory).set_mock_caller() for calling from script
        data, constraints_raw = load_excel(data_fmt='ecos', constraint_fmt='readable')

        print('Generating forecasts')

        forecaster = get_default_forecaster(n_lags)
        uncond_forecaster = UnconditionalForecaster(data, forecaster)
        uncond_forecaster.fit_covariance(n_resid, 'oasd')
        uncond_forecaster.fit()

        print('Generating constraints')

        state_space, conditional_constraints = convert_exog_to_constraint(data,
                                                                          uncond_forecaster.forecast_start.min() - 1)
        constraints = conditional_constraints + constraints_raw
        C, b = generate_constraint_mat_from_equations(constraints, state_space)

        print('Constraints compiled')
        start_date = uncond_forecaster.forecast_start.min() - 1
        y_hat = uncond_forecaster.y_hat.loc[start_date:].T.stack()
        y_exog = uncond_forecaster.df.loc[start_date:].T.stack()

        reconciler = Reconciler(y_hat, y_exog, uncond_forecaster.cov.cov_mat, C, b, lam)
        y_adj = reconciler.fit()

        data.update(y_adj, overwrite=False)

        df_out = data.reset_index().T.reset_index('variable').values.tolist()
        write_to_excel(df_out)

def main():
    print('Generating forecasts')

    forecaster = get_default_forecaster(n_lags)
    uncond_forecaster = UnconditionalForecaster(data, forecaster)
    uncond_forecaster.fit_covariance(n_resid, 'oasd')
    uncond_forecaster.fit()

    print('Generating constraints')

    state_space, conditional_constraints = convert_exog_to_constraint(data,
                                                                      uncond_forecaster.forecast_start.min() - 1)
    constraints = conditional_constraints + constraints_raw
    C, b = generate_constraint_mat_from_equations(constraints, state_space)

    print('Constraints compiled')
    start_date = uncond_forecaster.forecast_start.min() - 1
    y_hat = uncond_forecaster.y_hat.loc[start_date:].T.stack()
    y_exog = uncond_forecaster.df.loc[start_date:].T.stack()

    reconciler = Reconciler(y_hat, y_exog, uncond_forecaster.cov.cov_mat, C, b, lam)
    y_adj = reconciler.fit()

    data.update(y_adj, overwrite=False)

    df_out = data.reset_index().T.reset_index('variable').values.tolist()

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
    import os
    from mff.utils import load_synthetic_data

    os.chdir(r'C:\Users\dbilgin\OneDrive - International Monetary Fund (PRD)\prototype\mff')
    Tin = 1
    lam = 1e2
    n_lags = 4
    n_resid = 2
    cov_matrix_calc = 'oasd'

    use_excel = False
    directory = r'./data/input.xlsm'

    print('Reading data')
    if use_excel:
        wb = xw.Book(directory).set_mock_caller()
        data, constraints_raw = load_excel(directory, data_fmt='ecos', constraint_fmt='readable')
    else:
        data, constraints_raw = load_synthetic_data()

    print('Generating forecasts')

    forecaster = get_default_forecaster(n_lags)
    uncond_forecaster = UnconditionalForecaster(data, forecaster)
    uncond_forecaster.fit_covariance(n_resid, 'oasd')
    uncond_forecaster.fit()

    print('Generating constraints')

    state_space, conditional_constraints = convert_exog_to_constraint(data, uncond_forecaster.forecast_start.min() - 1)
    constraints = conditional_constraints + constraints_raw
    C, b = generate_constraint_mat_from_equations(constraints, state_space)

    print('Constraints compiled')
    start_date = uncond_forecaster.forecast_start.min() - 1
    y_hat = uncond_forecaster.y_hat.loc[start_date:].T.stack()
    y_exog = uncond_forecaster.df.loc[start_date:].T.stack()

    reconciler = Reconciler(y_hat, y_exog, uncond_forecaster.cov.cov_mat, C, b, lam)
    y_adj = reconciler.fit()

    data.update(y_adj, overwrite=False)

    df_out = data.reset_index().T.reset_index('variable').values.tolist()

    if use_excel:
        write_to_excel(df_out)

    # err = np.abs(y_adj['y'] - y_adj.drop('y', axis=1).sum(axis=1)).mean()
    # print(f'Avg reconcilation error for GDP accounting: {err}')
    #
    # y_agg = y_adj.groupby(['freq', 'year']).mean()
    # err = (y_agg.loc['q'] - y_agg.loc['a']).mean().mean()
    # print(f'Avg reconciliation error for quarters: {err}')
