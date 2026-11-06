"""Reproduce the GDP manuscript figure or the fiscal walkthrough figure.

Run from the repository root after installing the package:
    python paper/reproduce_figures.py
    python paper/reproduce_figures.py --example fiscal
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
from sktime.forecasting.compose import DirectReductionForecaster, make_reduction

from macroframe_forecast import MFF
from macroframe_forecast.examples import generate_example_GDP_df, generate_example_fiscal_df


def plot_gdp(data, first, final):
    """Plot GDP levels and growth, calculating growth before selecting years."""
    levels = pd.DataFrame({"First step": first.GDP, "Reconciled": final.GDP, "Historical": data.GDP})
    growth = levels.pct_change(fill_method=None) * 100
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    for ax, frame, title, units in zip(
        axes, [levels, growth], ["(a) GDP levels", "(b) Annual GDP growth"], ["Trillion USD", "Percent"]
    ):
        ax.plot(frame["First step"].loc[2024:], "--o", markersize=3, label="First step")
        ax.plot(frame["Reconciled"].loc[2024:], "-.", label="Reconciled")
        ax.plot(frame["Historical"].loc[2021:2024], color="red", label="Historical")
        ax.set(title=title, xlabel="Year", ylabel=units, xlim=(2021, 2031))
        ax.set_xticks([2022, 2024, 2026, 2028, 2030])
    axes[1].plot(2030, 4, "v", color="black", label="2030 target: 4%")
    for ax in axes:
        ax.legend(fontsize=8)
    return fig


def plot_fiscal(data, first, final):
    """Plot fiscal forecasts and the balance implied by each forecast stage."""
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
            ax.plot(data.pb.loc[2025:2030], "o", color="green", markersize=4, label="Required balance")
        if column == "exp":
            ax.plot(2030, 37, "v", color="black", label="2030 target")
        ax.set(title=title, xlabel="Year", ylabel="% of GDP", xlim=(2021, 2031))
        ax.set_xticks([2022, 2024, 2026, 2028, 2030])
        ax.legend(fontsize=8)
    return fig


def main():
    """Fit the selected example, verify its constraints, and save its outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=("gdp", "fiscal"), default="gdp")
    parser.add_argument("--output-dir", type=Path, default=Path("build/paper-figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.example == "gdp":
        data = generate_example_GDP_df()
        forecaster = make_reduction(LinearRegression(), strategy="direct", window_length=6)
        constraints = ["GDP_2030 - 1.04 * GDP_2029"]
    else:
        data = generate_example_fiscal_df()
        forecaster = DirectReductionForecaster(LinearRegression())
        constraints = ["pb_? - rev_? + exp_? - int_payments_?", "exp_2030 - 37"]
    model = MFF(
        data,
        forecaster=forecaster,
        equality_constraints=constraints,
        parallelize=False,
    )
    model.fit()
    first, final = model.df1, model.df2
    forecast_years = slice(2025, 2030)
    if args.example == "gdp":
        growth = pd.DataFrame({"First step": first.GDP, "Reconciled": final.GDP}).pct_change(fill_method=None) * 100
        np.testing.assert_allclose(growth.loc[2030, "Reconciled"], 4, rtol=0, atol=1e-8)
        print(growth.loc[forecast_years].round(4).to_string())
    else:
        residual = final.pb - final.rev + final.exp - final.int_payments
        np.testing.assert_allclose(residual.loc[forecast_years], 0, rtol=0, atol=1e-8)
        np.testing.assert_allclose(final.loc[2030, "exp"], 37, rtol=0, atol=1e-8)
        print(final.loc[forecast_years].round(4).to_string())
        print(f"Maximum forecast accounting residual: {residual.loc[forecast_years].abs().max():.3g}")
    np.testing.assert_allclose(final.to_numpy()[data.notna()], data.to_numpy()[data.notna()], rtol=0, atol=0)
    if not np.isfinite(final.to_numpy()).all():
        raise ValueError("The reconciled forecasts contain missing or non-finite values.")

    with plt.rc_context({"font.size": 10}):
        plot = plot_gdp if args.example == "gdp" else plot_fiscal
        fig = plot(data, first, final)
        for suffix in ("pdf", "png"):
            fig.savefig(args.output_dir / f"{args.example}_forecasts.{suffix}", dpi=180)
        plt.close(fig)

    pd.concat({"input": data, "first_step": first, "reconciled": final}, axis=1).to_csv(
        args.output_dir / f"{args.example}_forecasts.csv", index_label="year"
    )
    packages = ("macroframe-forecast", "numpy", "pandas", "scipy", "scikit-learn", "sktime", "matplotlib")
    environment = [f"Python {platform.python_version()}", *[f"{name}=={version(name)}" for name in packages]]
    (args.output_dir / f"{args.example}_environment.txt").write_text("\n".join(environment) + "\n", encoding="utf-8")
    print(f"Outputs saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
