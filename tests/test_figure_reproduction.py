"""Verify that the JOSS paper figure reproduction command works."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.test_support import check_gdp_forecasts

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("example", ["gdp", "fiscal"])
def test_figure_reproduction(tmp_path, example):
    """The standalone command saves usable figures and the verified forecasts."""
    output = tmp_path / "figures"
    command = [sys.executable, str(ROOT / "paper/reproduce_figures.py"), "--output-dir", str(output)]
    if example != "gdp":
        command += ["--example", example]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    for name in [
        f"{example}_forecasts.pdf",
        f"{example}_forecasts.png",
        f"{example}_forecasts.csv",
        f"{example}_environment.txt",
    ]:
        assert (output / name).stat().st_size > 0

    saved = pd.read_csv(output / f"{example}_forecasts.csv", header=[0, 1], index_col=0)
    assert set(saved.columns.get_level_values(0)) == {"input", "first_step", "reconciled"}
    final = saved["reconciled"]
    assert np.isfinite(final.to_numpy()).all()
    pd.testing.assert_frame_equal(final.loc[:2024], saved["input"].loc[:2024])
    if example == "gdp":
        check_gdp_forecasts(saved["input"], saved["first_step"], final)
    else:
        np.testing.assert_allclose(final.loc[2030, "exp"], 37.0, rtol=0, atol=1e-8)
        np.testing.assert_allclose(
            final.loc[2025:2030, "rev"] - final.loc[2025:2030, "exp"] + final.loc[2025:2030, "int_payments"],
            saved["input"].loc[2025:2030, "pb"],
            rtol=0,
            atol=1e-8,
        )
    environment = (output / f"{example}_environment.txt").read_text()
    assert "Python " in environment and "macroframe-forecast==" in environment
