# Reproducing Figure 1

[`reproduce_figures.py`](reproduce_figures.py) generates the JOSS paper's two-panel
US GDP forecasting figure from bundled data. It includes all required imports
and verifies the terminal growth target and preservation of historical values.
No data download is needed.

From the repository root, using Python 3.11 or 3.12:

```sh
python -m pip install -e .
python paper/reproduce_figures.py
```

Outputs are saved in `build/paper-figures/`:

- `gdp_forecasts.pdf`: the figure used in the manuscript.
- `gdp_forecasts.png`: a preview.
- `gdp_forecasts.csv`: input data, first-step forecasts, and reconciled forecasts.
- `gdp_environment.txt`: Python and numerical/plotting dependency versions.

The manuscript is maintained on `joss_branch`. When working on that branch,
update its figure after reviewing the output:

```sh
cp build/paper-figures/gdp_forecasts.pdf paper/gdp_forecasts.pdf
```

Use `--output-dir PATH` to choose another destination. Model estimates can vary
with dependency versions; retain the environment record with each run.

A narrative [GDP walkthrough](../docs/source/examples.rst#gdp-forecasting-with-a-growth-target)
explains the model, results, and growth constraint.

## GDP data and model

`generate_example_GDP_df()` supplies annual US nominal GDP for 1950-2030 from
the April 2025 IMF World Economic Outlook example, in trillions of US dollars.
Values through 2024 are observed; 2025-2030 are forecast.

The first stage uses ordinary least squares to fit a separate regression for
each forecast horizon, with the six most recent GDP observations as predictors:

```python
from sklearn.linear_model import LinearRegression
from sktime.forecasting.compose import make_reduction

forecaster = make_reduction(
    LinearRegression(), strategy="direct", window_length=6,
)
```

This is direct autoregressive forecasting, with coefficients estimated separately
for each horizon. The first-stage forecasts are unconstrained. MFF then applies
`GDP_2030 - 1.04 * GDP_2029`, an illustrative, user-specified target of 4 percent
nominal GDP growth in 2030.
The second stage uses the default five forecast-error windows, OAS covariance
shrinkage, and empirically estimated smoothing. No smoothing parameter is supplied.

The figure shows GDP levels and annual percentage growth. Growth is calculated
from the complete level series before selecting the plotting window, so 2025
growth uses the observed 2024 GDP value.

## Fiscal walkthrough figure

The same script also reproduces the multivariable
[fiscal walkthrough](../docs/source/examples.rst#fiscal-forecasting-walkthrough):

```sh
python paper/reproduce_figures.py --example fiscal
```

This writes `fiscal_forecasts.pdf`, `fiscal_forecasts.png`,
`fiscal_forecasts.csv`, and `fiscal_environment.txt` to the output directory.

`generate_example_fiscal_df()` supplies annual US fiscal data for 2001-2030 from
the April 2025 IMF World Economic Outlook example. All variables are percentages
of GDP. Expenditure (`exp`), revenue (`rev`), and interest payments
(`int_payments`) are observed through 2024 and forecast for 2025-2030. Primary
balance (`pb`) retains its prescribed path throughout the forecast horizon.

The fiscal first stage uses `DirectReductionForecaster(LinearRegression())`,
an OLS model, with serial execution. The second stage uses the same default
weighting and smoothing settings as the GDP example. It imposes:

```python
["pb_? - rev_? + exp_? - int_payments_?", "exp_2030 - 37"]
```

The primary-balance panel plots `rev - exp + int_payments` from each stage,
allowing comparison with the required balance. The script checks the identity
for all six forecast years, expenditure of 37% of GDP in 2030, preservation of
all supplied values, and finite reconciled forecasts.

