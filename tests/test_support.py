"""Shared assertions for documentation and paper reproduction tests."""

import numpy as np


def check_gdp_forecasts(data, first, final):
    """Independently check the direct OLS forecasts and reconciliation claims."""
    history = data.GDP.dropna().to_numpy()
    for horizon in range(1, 7):
        origins = range(5, len(history) - horizon)
        predictors = np.array([history[t - 5 : t + 1][::-1] for t in origins])
        targets = np.array([history[t + horizon] for t in origins])
        design = np.column_stack([np.ones(len(predictors)), predictors])
        coefficients = np.linalg.lstsq(design, targets, rcond=None)[0]
        expected = np.r_[1.0, history[-6:][::-1]] @ coefficients
        np.testing.assert_allclose(first.loc[2024 + horizon, "GDP"], expected, rtol=0, atol=1e-8)

    np.testing.assert_allclose(final.loc[2030, "GDP"], 1.04 * final.loc[2029, "GDP"], rtol=0, atol=1e-8)
    first_curvature = np.square(np.diff(first.GDP.loc[2025:2030], n=2)).sum()
    final_curvature = np.square(np.diff(final.GDP.loc[2025:2030], n=2)).sum()
    assert final_curvature < first_curvature
