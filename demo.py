import sys
import xlwings as xw

sys.path.append(r'mff')

from mff.ecos_reader import load_excel
from mff.default_forecaster import get_default_forecaster


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


def run(lam, n_resid=5, n_lags=4, cov_matrix_calc='oasd'):
    print('Reading data')
    with xw.App(visible=False) as app:
        wb = xw.Book.caller()  # xw.Book(directory).set_mock_caller() for calling from script
        data, constraints_raw = load_excel(data_fmt='ecos', constraint_fmt='readable')

        print('Generating forecasts')
        forecaster = get_default_forecaster(n_lags)
        mff = MFF(data, constraints_raw, lam, forecaster, n_resid=n_resid, cov_calc_method=cov_matrix_calc)
        mff.fit()

        data.update(mff.y_reconciled, overwrite=False)

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
    import os
    from mff.utils import load_synthetic_data
    from mff.mff import MFF

    os.chdir(r'C:\Users\dbilgin\OneDrive - International Monetary Fund (PRD)\prototype\mff')
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

    forecaster = get_default_forecaster(n_lags)
    mff = MFF(data, constraints_raw, lam=lam, forecaster=forecaster, n_resid=n_resid, cov_calc_method=cov_matrix_calc)
    mff.fit()

    data.update(mff.y_reconciled, overwrite=False)
    df_out = data.reset_index().T.reset_index('variable').values.tolist()
    if use_excel:
        write_to_excel(df_out)

    # err = np.abs(y_adj['y'] - y_adj.drop('y', axis=1).sum(axis=1)).mean()
    # print(f'Avg reconcilation error for GDP accounting: {err}')
    #
    # y_agg = y_adj.groupby(['freq', 'year']).mean()
    # err = (y_agg.loc['q'] - y_agg.loc['a']).mean().mean()
    # print(f'Avg reconciliation error for quarters: {err}')
