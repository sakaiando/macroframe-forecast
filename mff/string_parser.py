import re
import pandas as pd
import sympy as sp
import numpy as np
from itertools import product
import string



def find_strings_to_replace_wildcard(constraint, variables, wildcard):
    # utilize sympy to extract variables from symbolic equations
    varlist_with_wildcard = ['^' + str(v).replace(wildcard, '(.*)') + '$'
                             for v in sp.sympify(constraint).free_symbols]
    missing_string_set_list = []
    for w in varlist_with_wildcard:
        missing_string = []
        for v in variables:
            match = re.compile(w).search(v)
            if match:
                missing_string.append(match.group(1))
        missing_string_set_list.append(set(missing_string))
    missing_string_list = set.intersection(*missing_string_set_list)
    return missing_string_list


def expand_wildcard(constraints_with_alphabet_wildcard, variables, wildcard):
    extended_constraints = []
    for constraint in constraints_with_alphabet_wildcard:
        if wildcard not in constraint:
            extended_constraints.append(constraint)
        else:
            missing_string_list = find_strings_to_replace_wildcard(constraint, variables, wildcard)
            extended_constraints += [constraint.replace(f'{wildcard}', m) for m in missing_string_list]
    return extended_constraints


def find_permissible_wildcard(constraints_with_wildcard, size_of_candidates=4):
    # string a,b,...,aa,ab,...,zzzz
    wildcard_candidates = [''.join(p) for n in range(1, size_of_candidates + 1)
                           for p in product(string.ascii_lowercase, repeat=n)]

    for w in wildcard_candidates:
        if len(w) == len(set(w)):  # skip wildcards with repeated letters (e.g. avoid filling a? wildcard with aaa, which would be ambigous)
            if w not in ''.join(constraints_with_wildcard):
                return w
    raise RuntimeError('Failed to find a unique sequence.'
                       'Consider increasing max_len or revising variable names.')


def generate_constraints_from_equations(constraints_list, variables_list, wildcard_string='?'):
    # add error message to say variables_list cannot include *
    wildcard_temp = find_permissible_wildcard(constraints_list)
    # replace wildcard with alphabet to utilize sympy
    constraints_with_alphabet_wildcard = [c.replace(wildcard_string, wildcard_temp) for c in constraints_list]
    constraints = expand_wildcard(constraints_with_alphabet_wildcard=constraints_with_alphabet_wildcard,
                                  variables=variables_list,
                                  wildcard=wildcard_temp,
                                  )

    A, b = sp.linear_eq_to_matrix(constraints, sp.sympify(variables_list))

    A = np.array(A).astype(float)
    b = np.array(b).astype(float)

    idx = statespace_str_to_multiindex(variables_list)
    dfA = pd.DataFrame(A, index=constraints, columns=idx)
    dfB = pd.DataFrame(b, index=constraints, columns=['constant'])['constant']

    return dfA, dfB


def split_string(input_str):
    # Regular expression to match the pattern
    pattern = re.compile(r'([a-zA-Z0-9_]+)_(\d+)([A-Z]*)(\d*)')

    match = pattern.match(input_str)

    if not match:
        raise ValueError("Input string does not match the expected pattern.")

    prefix = match.group(1)
    year = int(match.group(2))
    period_letter = match.group(3) if match.group(3) else 'A'
    period_number = int(match.group(4)) if match.group(4) else 1

    return prefix, year, period_letter, period_number


def statespace_str_to_multiindex(idx_str):
    idx_split = [split_string(i) for i in idx_str]
    idx_multi = pd.MultiIndex.from_tuples(idx_split, names=['variable', 'year', 'freq', 'subperiod'])
    return idx_multi


if __name__ == '__main__':
    constraints_with_wildcard = ['a1_2022 - a2_2023',
                                 'a1? + a2? - 1',
                                 '?2022 + ?2023 - 100',
                                 '? - (?Q1 + 3*?Q2 + ?Q3 + ?Q4)']

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

    variables = variables_a + variables_q

    A, b = generate_constraints_from_equations(constraints_with_wildcard,
                                               variables,
                                               wildcard_string='?',
                                               )

    print("Matrix A:\n", A)
    print("Matrix b:\n", b)


#    import mff.mff.tests.test_string_parser as tests