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