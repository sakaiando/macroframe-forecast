"""Regression checks for the documented constraint syntax."""

import numpy as np
import pandas as pd
import pytest
from sympy.solvers.solveset import NonlinearError

from macroframe_forecast.utils import OrganizeCells, StringToMatrixConstraints


def test_expression_only_inequality_constraints():
    df = pd.DataFrame(
        {"a": [1.0, np.nan, np.nan]},
        index=pd.period_range("2020", periods=3, freq="Y"),
    )
    df0, all_cells, unknown_cells, known_cells, _ = OrganizeCells(df)
    stacked = df0.T.stack(future_stack=True)

    coefficients, bounds = StringToMatrixConstraints(
        stacked, all_cells, unknown_cells, known_cells, ["a? - 5"]
    )

    np.testing.assert_array_equal(coefficients, np.eye(2))
    np.testing.assert_array_equal(bounds, [[5.0], [5.0]])

    with pytest.raises(NonlinearError):
        StringToMatrixConstraints(
            stacked, all_cells, unknown_cells, known_cells, ["a? - 5 <= 0"]
        )
