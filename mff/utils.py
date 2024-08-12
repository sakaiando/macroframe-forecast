import numpy as np
import pandas as pd
from itertools import product


def multiindex_from_multiindex_product(idx_left, idx_right):
    concat_list = []
    left_is_multiindex = isinstance(idx_left, pd.MultiIndex)
    right_is_multiindex = isinstance(idx_right, pd.MultiIndex)
    for a, b in product(idx_left, idx_right):
        if left_is_multiindex and right_is_multiindex:
            item = (*a, *b)
        elif left_is_multiindex and not right_is_multiindex:
            item = (*a, b)
        elif not left_is_multiindex and right_is_multiindex:
            item = (a, *b)
        else:
            item = (a, b)
        concat_list.append(item)
    return pd.MultiIndex.from_tuples(concat_list, names=list(idx_left.names) + list(idx_right.names))


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
                  [-.3, np.nan, np.nan, 1.1, 3.60],
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