import pandas as pd
import xlwings as xw

ECOS_COLS_TO_DROP = ['Frequency', 'Scale', 'Display_scale']


def drop_ecos_metadata(df):
    cols_to_drop = df.index.intersection(ECOS_COLS_TO_DROP)
    df.index.name = 'year'
    df.columns.name = 'variable'
    df = df.drop(cols_to_drop).astype(float)
    df.index = df.index.astype(int)
    return df


def split_index_to_subfreq(idx):
    idx = idx.astype(str).str.split('([a-zA-Z])', expand=True)
    if len(idx.shape) == 1:  # if only annual is present
        idx = idx + 'A1'
        idx = idx.str.split('([a-zA-Z])', expand=True)
    return idx


def process_split_index(idx):
    idx = pd.DataFrame.from_records(idx, columns=['year', 'freq', 'subperiod'])
    idx['freq'] = idx['freq'].fillna('A')
    idx['subperiod'] = idx['subperiod'].fillna(1)
    idx = idx.astype({'year': int, 'subperiod': int})
    idx = pd.MultiIndex.from_frame(idx)
    return idx


def process_ecos_df(df):
    df = drop_ecos_metadata(df)

    idx = split_index_to_subfreq(df.index)
    idx = process_split_index(idx)

    df.index = idx
    return df


def read_excel_sheet_while_open(sheet_name):
    wb = xw.Book().caller()
    value = wb.sheets[sheet_name].used_range.value
    return value


def create_dataframe_with_index_and_columns(data):
    """
    Creates a pandas DataFrame where the first row is used as the column names
    and the first column is used as the index.

    Parameters:
    data (list of list): The input 2D array with the first row as columns
                         and the first column as index.

    Returns:
    pd.DataFrame: The constructed DataFrame.
    """
    # Extract the columns from the first row (excluding the first element)
    columns = data[0][1:]

    # Extract the index from the first column (excluding the first element)
    index = [row[0] for row in data[1:]]

    # Extract the actual data (excluding the first row and first column)
    data_values = [row[1:] for row in data[1:]]

    # Create the DataFrame
    df = pd.DataFrame(data=data_values, index=index, columns=columns)

    return df

def load_excel_ecos(dir):
    value = read_excel_sheet_while_open('data')
    data = create_dataframe_with_index_and_columns(value).T
    data = process_ecos_df(data)
    return data


def load_constraints_readable(dir):
    constraints = read_excel_sheet_while_open('constraints')
    return constraints


def load_constraints_matrix(dir):
    constraints_raw = pd.read_excel(dir, sheet_name='constraints', header=list(range(0, 3)), index_col=0)
    return constraints_raw


def load_excel_correct_fmt(dir):
    data = pd.read_excel(dir, index_col=0, header=list(range(0, 3)), sheet_name='data').T
    return data


def load_excel(dir=None, data_fmt='correct', constraint_fmt='readable'):
    data_loader = {'ecos': load_excel_ecos,
                   'correct': load_excel_correct_fmt,
                   }
    data = data_loader[data_fmt](dir)
    data = reshape_data(data)


    constraint_loader = {'readable': load_constraints_readable,
                         'matrix': load_constraints_matrix,
                         }
    constraints_raw = constraint_loader[constraint_fmt](dir)

    return data, constraints_raw


def reshape_data(df):
    df.columns.name = 'variable'
    df = df.unstack(['freq', 'subperiod'])
    return df


if __name__ == '__main__':
    directory = r'./data/input.xlsx'
    xw.Book(directory).set_mock_caller()
    d, C = load_excel(directory, data_fmt='ecos', constraint_fmt='readable')
