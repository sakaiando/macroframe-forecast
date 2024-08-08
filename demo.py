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
    d = np.array([[4.19269782, 4.50827723, -0.31557941, 0.94078695, 6.20608082],
                  [2.68600667, 2.38248276, 0.30352391, 1.01345659, 7.87206013],
                  [-9.6666219, -10.21580561, 0.54918371, 0.98303537, 6.94096329],
                  [-8.80029164, -8.98838301, 0.18809136, 0.89659413, 4.44459],
                  [-9.57310499, -10.25879872, 0.68569373, 0.85525298, 4.07604537],
                  [-4.46607605, -4.12322211, -0.34285394, 0.73004356, -0.1059688],
                  [-3.08267219, -2.27094799, -0.8117242, 0.65473971, 1.16665966],
                  [-7.97944828, -7.6477212, -0.33172708, 0.62331964, 3.25382766],
                  [-7.60349268, -6.85877335, -0.74471932, 0.666698, 4.50951225],
                  [-4.8933913, -1.53269505, -3.36069625, 0.8217316, 5.49918132],
                  [-5.84952694, -2.30867284, -3.54085409, 0.93535558, 5.27867269],
                  [-7.2865745, -4.11339873, -3.17317577, 0.97287564, 6.62366768],
                  [-7.02579334, -3.45242575, -3.57336759, 1.0200627, 8.49323783],
                  [-4.79978484, -0.6074836, -4.19230124, 1.22284029, 10.83206173],
                  [-6.35798996, -2.30787231, -4.05011765, 1.4166728, 5.57449392],
                  [-3.44345547, -1.04703138, -2.39642409, 1.39326793, -5.45540743],
                  [-4.62938995, -1.06333164, -3.56605831, 1.32679943, 6.71661007],
                  [-4.87156949, -0.42571034, -4.44585915, 1.39171031, 2.67141158],
                  [0.92859121, 3.97451703, -3.04592582, 1.28560226, 1.31895589],
                  [1.85091714, 4.55292324, -2.7020061, 1.32813739, 0.63278433],
                  [1.14007539, 3.84181176, -2.70173637, 1.32884317, 2.69692121],
                  [-2.08309413, 1.1554302, -3.23852432, 1.10962523, 5.16740584],
                  [-2.73351939, 2.01476155, -4.74828094, 1.10659767, 1.94393587],
                  [-1.91118686, 1.74737421, -3.65856107, 1.12928253, 2.93804027],
                  [-2.19561478, 0.76673498, -2.96234976, 1.18149004, 4.03027463],
                  [-3.34926406, 0.08842557, -3.43768963, 1.11959806, 2.51130642],
                  [0.55166731, 2.10660931, -1.554942, 1.14128186, -3.33537417],
                  [-2.45911205, 0.03341456, -2.49252661, 1.1835266, 4.78908793],
                  [-2.72867408, -0.26952854, np.nan, 1.05387707, 4.51088288],
                  [-2.98144077, -0.55254428, np.nan, 1.1, 4.25537143],
                  [np.nan, np.nan, np.nan, 1.1, 4.04312423],
                  [np.nan, np.nan, np.nan, 1.1, 3.8694549],
                  [np.nan, np.nan, np.nan, 1.1, 3.72591658],
                  [np.nan, np.nan, np.nan, 1.1, 3.60422314],
                  [np.nan, np.nan, np.nan, 1.1, 3.49732526],
                  [np.nan, np.nan, np.nan, 1.1, 3.39836701],
                  [np.nan, np.nan, np.nan, 1.1, 3.30188283]])
    idx = pd.Int64Index([1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004,
            2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
            2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026,
            2027, 2028, 2029, 2030],
           dtype='int64', name='year')
    col = pd.MultiIndex.from_tuples([( 'CA', 'A', 1),
            ( 'TA', 'A', 1),
            ( 'IA', 'A', 1),
            ( 'FX', 'A', 1),
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

    #y_hat.update(y_exog)
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
