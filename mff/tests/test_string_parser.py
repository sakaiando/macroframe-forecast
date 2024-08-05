import pytest
import sys
sys.path.append(r'/mff')

import numpy as np
import pandas as pd
from itertools import product

from mff.mff.string_parser import generate_constraint_mat_from_equations


def unit_test1():
    constraints = ['a1_2022 - a2_2023',
                   f'a1? + a2? - 1',
                   f'?2022 + ?2023 - 100',
                   f'? - (?Q1 + 3*?Q2 + ?Q3 + ?Q4)']

    dfA = pd.DataFrame({'a1': np.random.rand(5),
                        'a2': np.random.rand(5),
                        'x': np.random.rand(5),
                        'z': np.random.rand(5)},
                       index=pd.period_range(start='2021',
                                             end='2025',
                                             freq='A',
                                             )
                       )
    dfQ = pd.DataFrame({'q1': np.random.rand(20),
                        'q2': np.random.rand(20),
                        'x': np.random.rand(20),
                        'y': np.random.rand(20)},
                       index=pd.period_range(start='2021Q1',
                                             end='2025Q4',
                                             freq='Q',
                                             )
                       )

    variables_a = [f'{a}_{b}' for a, b in product(dfA.columns, dfA.index)]
    variables_q = [f'{a}_{b}' for a, b in product(dfQ.columns, dfQ.index)]

    ss = variables_a + variables_q

    A, b = generate_constraint_mat_from_equations(constraints, ss)

    import os

    os.chdir(r'C:\Users\dbilgin\OneDrive - International Monetary Fund (PRD)\prototype\mff\mff')
    A_test = pd.read_csv(r'tests\constraint_A.csv', header=0, index_col=0)
    b_test = pd.read_csv(r'tests\constraint_b.csv', index_col=0)['constant']

    assert (A_test == A).all().all()
    assert (b_test == b).all()

    print('Unit test 1 passed')


unit_test1()
