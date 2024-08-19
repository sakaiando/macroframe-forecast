import re
import pandas as pd
import sympy as sp
import numpy as np
from itertools import product
import string



def find_strings_to_replace_wildcard(constraint, variables, wildcard):
    """
    Utilize sympy to extract variables from constraint, find wildcard to fill.

    Args:
        constraint (str): The constraint equation string containing the wildcard.
        variables (list): List of variable names.
        wildcard (str): The wildcard character to be replaced.

    Returns:
        set: A set of strings that can replace the wildcard.
    """
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
    """
       Expand constraints containing wildcards by replacing the wildcard with actual variable names.

       Args:
           constraints_with_alphabet_wildcard (list): List of constraints where wildcard is alphanumeric.
           variables (list): List of sympy variable names.
           wildcard (str): The wildcard string to be replaced.

       Returns:
           list: List of constraints with wildcards extended and replaced.
       """
    extended_constraints = []
    for constraint in constraints_with_alphabet_wildcard:
        if wildcard not in constraint:
            extended_constraints.append(constraint)
        else:
            missing_string_list = find_strings_to_replace_wildcard(constraint, variables, wildcard)
            extended_constraints += [constraint.replace(f'{wildcard}', m) for m in missing_string_list]
    return extended_constraints


def find_permissible_wildcard(constraints_with_wildcard, size_of_candidates=4):
    """
        Find a permissible wildcard string that does not conflict with existing variable names by cycling over sequences
        a, b, ..., z, ab, ac, ... zy, abc, abd, ...

        Args:
            constraints_with_wildcard (list): List of constraints containing wildcards.
            size_of_candidates (int): Maximum length of wildcard search.

        Returns:
            str: A permissible wildcard string.
        """
    wildcard_candidates = [''.join(p) for n in range(1, size_of_candidates + 1)
                           for p in product(string.ascii_lowercase, repeat=n)]

    for w in wildcard_candidates:
        if len(w) == len(set(w)):  # skip wildcards with repeated letters (e.g. avoid filling a? wildcard with aaa, which would be ambigous)
            if w not in ''.join(constraints_with_wildcard):
                return w
    raise RuntimeError('Failed to find a unique sequence.'
                       'Consider increasing size_of_candidates or revising variable names.')


def generate_constraint_mat_from_equations(constraints_list, variables_list, wildcard_string='?'):
    """
       Generate constraint matrix (A) and vector (b) from a list of constraint equations.

       Args:
           constraints_list (list): List of constraint equations.
           variables_list (list): List of variable names.
           wildcard_string (str): The wildcard character in the constraints.

       Returns:
           DataFrame, Series: Constraint matrix (A) and vector (b).
       """
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
    """
    Split a string into the variable name, date and subperiod components based on the pattern
    (variablename)_(year)(freq)(subperiod)

    Args:
        input_str (str): The input string to be split.

    Returns:
        tuple: Components of the split string.
    """
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
    """
    Convert a list of state space strings with format (variablename)_(year)(freq)(subperiod) to a MultiIndex.

    Args:
       idx_str (list): List of state space strings.

    Returns:
       MultiIndex: MultiIndex object.
    """
    idx_split = [split_string(i) for i in idx_str]
    idx_multi = pd.MultiIndex.from_tuples(idx_split, names=['variable', 'year', 'freq', 'subperiod'])
    return idx_multi


def find_first_na_in_df(df):
    """
      Find the first occurrence of NA in a DataFrame.

      Args:
          df (DataFrame): The input DataFrame.

      Returns:
          int: The index of the first NA.
      """
    forecast_start = df.isna().idxmax()[df.isna().any()].min()  # find first NA
    return forecast_start


def calculate_state_space(ss_idx):
    """
    Calculate the string state space from a MultiIndex with columns ['variable', 'year', 'freq', 'subperiod'].

    Args:
        ss_idx (MultiIndex): The input MultiIndex.

    Returns:
        list: List of state space strings.
    """
    ss = ss_idx.to_frame().astype(str)
    ss = ss['variable'] + '_' + ss['year'] + (ss['freq'] + ss['subperiod']).replace({'A1': ''})
    return ss.tolist()


def convert_exog_to_constraint(df, forecast_start=None):
    """
    Convert exogenous variables to constraints.

    Args:
        df (DataFrame): The input DataFrame.
        forecast_start (int, optional): The start index for forecasting. Defaults to None.

    Returns:
        tuple: State space strings and conditional forecast constraints.
    """
    if forecast_start is None:
        forecast_start = df.index.min()
    df_stacked = df.stack(df.columns.names, dropna=False)
    fcast_stacked = df_stacked[df_stacked.index.get_level_values('year') >= forecast_start]
    ss_str = calculate_state_space(fcast_stacked.index)

    fixed_fcasts = fcast_stacked.dropna()
    conditional_fcast_constraints = [f'{i} - {j}' for i, j in
                                     zip(calculate_state_space(fixed_fcasts.index), fixed_fcasts.values)]
    return ss_str, conditional_fcast_constraints


def generate_constraints(df, constraints_list, forecast_start=None, n_hist_points=2):
    """
    Generate constraints for the forecasting model. These are generated using 3 sources:
    - non-NA values of rows of df containing NAs
    - n_hist_points rows preceeding the first row containing an NA
    - list of strings in constraints_list

    Args:
        df (DataFrame): The input DataFrame containing historical data and forecast dates (indicated by rows containing NA).
        constraints_list (list): List of string constraints.
        forecast_start (int, optional): The start index for forecasting. Defaults to None, which finds the first row containing an NA.
        n_hist_points (int, optional): Number of historical points to pad constraints. Defaults to 2.

    Returns:
        tuple: Constraint matrix (C) and vector (b).
    """
    if forecast_start is None:
        forecast_start = find_first_na_in_df(df)
    state_space, conditional_constraints = convert_exog_to_constraint(df, forecast_start - n_hist_points)
    constraints = conditional_constraints + constraints_list
    C, b = generate_constraint_mat_from_equations(constraints, state_space)
    return C, b


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

    A, b = generate_constraint_mat_from_equations(constraints_with_wildcard, variables, wildcard_string='?')

    print("Matrix A:\n", A)
    print("Matrix b:\n", b)


#    import mff.mff.tests.test_string_parser as tests