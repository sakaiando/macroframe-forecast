import re

import pandas as pd
import sympy as sp
import numpy as np
from itertools import product
import fnmatch
import string


def extract_variables(eq: str, return_str=False):
    variables = sp.sympify(eq).free_symbols
    if return_str:
        variables = [str(v) for v in variables]
    return variables


def parse_equations_to_matrix(equations, state_space):
    # Extract all symbols from the equations
    variables = sp.sympify(state_space)
    A, b = sp.linear_eq_to_matrix(equations, variables)
    return A, b, variables


def expand_wildcards(constraints, state_space, wildcard):
    # create constraints from wildcards
    extended_constraints = []
    for constraint in constraints:
        short_varlist = extract_variables(constraint, return_str=True)
        short_varlist = [s.replace(wildcard, '*') for s in
                         short_varlist]  # replace user-specified with regex wildcard value

        long_varlist = []
        for v in short_varlist:
            long_varlist += fnmatch.filter(state_space, v)

        if short_varlist == long_varlist:
            extended_constraints.append(constraint)
        else:
            common_phrases = []
            for v_long in long_varlist:
                for v_short in short_varlist:
                    match = re.compile(v_short.replace('*', '(.*)')).search(v_long)
                    if match:
                        common_phrases.append(match.group(1))

            count_dict = {i: common_phrases.count(i) for i in common_phrases}
            common_phrases = [v for v in count_dict.keys() if count_dict[v] == max(count_dict.values())]
            extended_constraints += [constraint.replace(f'{wildcard}', common_phrase) for common_phrase in
                                     common_phrases]

    return extended_constraints


# Function to check if a sequence exists in the list of strings
def sequence_exists(sequence, constraints):
    for constraint in constraints:
        if sequence in constraint:
            return True
    return False


# Function to generate sequences of increasing length
def generate_sequences(length, letters):
    return (''.join(combination) for combination in product(letters, repeat=length))


# Function to iterate over sequences a, b .., aa, ab, ... aaa, aab of increasing length
def find_unique_sequence(constraints, max_len=4):
    letters = string.ascii_lowercase  # 'a' to 'z'
    length = 1  # Starting length of sequences

    while length <= max_len:
        for sequence in generate_sequences(length, letters):
            if not sequence_exists(sequence, constraints):
                return sequence
        length += 1  # Increase the length for the next iteration
    raise RuntimeError(f'Failed to find a unique sequence within the maximum length of {max_len}. '
                       f'Consider increasing max_len or revising variable names.')


def generate_constraints_from_equations(constraints_list, state_space, wildcard_string='?'):
    wildcard_temp = find_unique_sequence(constraints_list)
    constraints = [c.replace(wildcard_string, wildcard_temp) for c in constraints_list]

    constraints = expand_wildcards(constraints=constraints,
                                   state_space=state_space,
                                   wildcard=wildcard_temp,
                                   )

    A, b, variables = parse_equations_to_matrix(constraints, state_space)

    A = np.array(A).astype(float)
    b = np.array(b).astype(float)
    idx = pd.Index(constraints, name='constraints')
    dfA = pd.DataFrame(A, index=idx, columns=pd.Index([v.name for v in variables], name='variables'))
    dfB = pd.DataFrame(b, index=idx, columns=['constant'])['constant']

    return dfA, dfB


if __name__ == '__main__':
     import mff_files.mff.tests.test_string_parser as tests

