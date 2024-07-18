import pandas as pd

ECOS_COLS_TO_DROP = ['Frequency', 'Scale', 'Display_scale']


def drop_ecos_metadata(df):
    cols_to_drop = df.index.intersection(ECOS_COLS_TO_DROP)
    df.index.name = 'year'
    df.columns.name = 'variable'
    df = df.drop(cols_to_drop).astype(float)
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


def load_excel_ecos(dir):
    data = pd.read_excel(dir, index_col=0, header=0, sheet_name='data').T
    data = drop_ecos_metadata(data)
    data = process_ecos_df(data)
    return data


def load_constraints_readable(dir):
    constraints = pd.read_excel(dir, index_col=0, header=0, sheet_name='constraints')['constraint']
    return constraints


def load_constraints_matrix(dir):
    constraints_raw = pd.read_excel(dir, sheet_name='constraints', header=list(range(0, 3)), index_col=0)
    return constraints_raw


def load_excel_correct_fmt(dir):
    data = pd.read_excel(dir, index_col=0, header=list(range(0, 3)), sheet_name='data').T
    return data


def load_excel(dir, data_fmt='correct', constraint_fmt='readable'):
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
    d, C = load_excel(directory, data_fmt='ecos', constraint_fmt='readable')
