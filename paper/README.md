# Reproducing Figure 1

[`reproduce_figures.py`](reproduce_figures.py) generates the JOSS paper's four-panel
US fiscal forecasting figure from bundled data. It includes all required imports
and verifies the accounting identity and terminal expenditure target. No data
download is needed.

From the repository root, using Python 3.11 or later:

```sh
python -m pip install -e .
python paper/reproduce_figures.py
```

Outputs are saved in `build/paper-figures/`:

- `fiscal_forecasts.pdf`: the figure used in the manuscript.
- `fiscal_forecasts.png`: a preview.
- `fiscal_forecasts.csv`: input data, first-step forecasts, and reconciled forecasts.
- `environment.txt`: Python and numerical/plotting dependency versions.

To update the manuscript figure after reviewing the output:

```sh
cp build/paper-figures/fiscal_forecasts.pdf paper/fiscal_forecasts.pdf
```

Use `--output-dir PATH` to choose another destination. Model estimates can vary
with dependency versions; retain the environment record with each run.

A narrative [fiscal walkthrough](../docs/source/examples.rst#fiscal-forecasting-walkthrough)
explains the inputs, model, results, and constraint checks step by step.

## Data and model

`generate_example_fiscal_df()` supplies annual US fiscal data for 2001-2030 from
the April 2025 IMF World Economic Outlook example. All variables are percentages
of GDP. Expenditure (`exp`), revenue (`rev`), and interest payments
(`int_payments`) are observed through 2024 and forecast for 2025-2030. Primary
balance (`pb`) retains its prescribed path throughout the forecast horizon.

The first stage uses `DirectReductionForecaster(LinearRegression())`, an OLS
model, with serial execution. The second stage uses the default five
forecast-error windows, OAS covariance shrinkage, and empirically estimated
smoothing parameters. It imposes:

```python
["pb_? - rev_? + exp_? - int_payments_?", "exp_2030 - 37"]
```

The primary-balance panel plots `rev - exp + int_payments` from each stage,
allowing comparison with the required balance. The script checks the identity
for all six forecast years, expenditure of 37% of GDP in 2030, preservation of
all supplied values, and finite reconciled forecasts.

The simpler GDP example remains available in the
[documentation](https://sakaiando.github.io/macroframe-forecast/examples.html)
and through `generate_example_GDP_df()` in `macroframe_forecast.examples`.
