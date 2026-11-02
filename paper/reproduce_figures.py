"""Reproduce the fiscal forecasting figure in the JOSS paper.

Run from the repository root after installing the package:
    python paper/reproduce_figures.py
"""

import argparse
from importlib.metadata import version
from pathlib import Path
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sktime.forecasting.compose import DirectReductionForecaster

from macroframe_forecast import MFF
from macroframe_forecast.examples import generate_example_fiscal_df


def main():
    """Fit the fiscal example, verify its constraints, and save the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("build/paper-figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = generate_example_fiscal_df()
    model = MFF(
        data,
        forecaster=DirectReductionForecaster(LinearRegression()),
        equality_constraints=["pb_? - rev_? + exp_? - int_payments_?", "exp_2030 - 37"],
        parallelize=False,
    )
    model.fit()
    first, final = model.df1, model.df2
    forecast_years = slice(2025, 2030)
    residual = final.pb - final.rev + final.exp - final.int_payments
    np.testing.assert_allclose(residual.loc[forecast_years], 0, rtol=0, atol=1e-8)
    np.testing.assert_allclose(final.loc[2030, "exp"], 37, rtol=0, atol=1e-8)
    np.testing.assert_allclose(final.to_numpy()[data.notna()], data.to_numpy()[data.notna()], rtol=0, atol=0)
    if not np.isfinite(final.to_numpy()).all():
        raise ValueError("The reconciled forecasts contain missing or non-finite values.")

    # The primary-balance panel shows the balance implied by each forecast,
    # rather than the supplied pb column, which is unchanged in both stages.
    with plt.rc_context({"font.size": 10}):
        fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout="constrained")
        panels = [
            ("exp", "(a) Expenditure"),
            ("rev", "(b) Revenue"),
            ("int_payments", "(c) Interest payments"),
            ("pb", "(d) Implied primary balance"),
        ]
        for ax, (column, title) in zip(axes.flat, panels):
            for frame, label, style in [(first, "First step", "--"), (final, "Reconciled", "-.")]:
                series = frame[column] if column != "pb" else frame.rev - frame.exp + frame.int_payments
                ax.plot(series.loc[2024:], style, label=label)
            ax.plot(data[column].loc[2021:2024], color="red", label="Historical")
            if column == "pb":
                ax.plot(data.pb.loc[forecast_years], "o", color="green", markersize=4, label="Required balance")
            if column == "exp":
                ax.plot(2030, 37, "v", color="black", label="2030 target")
            ax.set(title=title, xlabel="Year", ylabel="% of GDP", xlim=(2021, 2031))
            ax.set_xticks([2022, 2024, 2026, 2028, 2030])
            ax.legend(fontsize=8)
        for suffix in ("pdf", "png"):
            fig.savefig(args.output_dir / f"fiscal_forecasts.{suffix}", dpi=180)
        plt.close(fig)

    pd.concat({"input": data, "first_step": first, "reconciled": final}, axis=1).to_csv(
        args.output_dir / "fiscal_forecasts.csv", index_label="year"
    )
    packages = ("macroframe-forecast", "numpy", "pandas", "scipy", "scikit-learn", "sktime", "matplotlib")
    environment = [f"Python {platform.python_version()}", *[f"{name}=={version(name)}" for name in packages]]
    (args.output_dir / "environment.txt").write_text("\n".join(environment) + "\n", encoding="utf-8")
    print(final.loc[forecast_years].round(4).to_string())
    print(f"Maximum forecast accounting residual: {residual.loc[forecast_years].abs().max():.3g}")
    print(f"Outputs saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
