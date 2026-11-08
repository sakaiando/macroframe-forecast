# `macroframe-forecast`: a Python package to assist with macroframework forecasting


[![!pypi](https://img.shields.io/pypi/v/macroframe-forecast?color=green)](https://pypi.org/project/macroframe-forecast/) [![Downloads](https://static.pepy.tech/personalized-badge/macroframe-forecast?period=total&units=international_system&left_color=grey&right_color=blue&left_text=cumulative%20(pypi))](https://pepy.tech/project/macroframe-forecast)

This package assists forecasting with constraints and is based on the following papers:
* [A Python Package to Assist Macroframework Forecasting: Concepts and Examples (2025)](https://www.imf.org/en/Publications/WP/Issues/2025/08/29/A-Python-Package-to-Assist-Macroframework-Forecasting-Concepts-and-Examples-570041).
* [Smooth Forecast Reconciliation (2024)](https://www.imf.org/en/Publications/WP/Issues/2024/03/22/Smooth-Forecast-Reconciliation-546654)
* [Systematizing Macroframework Forecasting: High-Dimensional Conditional Forecasting with Accounting Identities (2023)](https://link.springer.com/article/10.1057/s41308-023-00225-8)

The targeted audience is policy makers that conduct multivariate economic forecasting with large *N* and small *T* (many time series and short sample periods), but the methodology can also be applied to other problems.

# Documentation

Please refer to [this link](https://sakaiando.github.io/macroframe-forecast/) for documentation.

# Installation

Python 3.11 and 3.12 are the supported, tested versions. Python 3.13 and later
are outside the supported range; see [PR #80](https://github.com/sakaiando/macroframe-forecast/pull/80)
for the corresponding runtime compatibility and package metadata changes.
To install the `macroframe-forecast` package, run:

```shell
pip install macroframe-forecast
```

# Quick start

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from macroframe_forecast import MFF

# true data
rng = np.random.default_rng(0)
df_true = pd.DataFrame({
    'var1': rng.normal(size=30),
    'var2': rng.normal(size=30),
}, index=pd.period_range("2000", periods=30, freq="Y"))
df_true['sum'] = df_true['var1'] + df_true['var2']

# input dataframe
df = df_true.copy()
fh = 5
df.iloc[-fh:, 1:] = np.nan

# apply MFF
m = MFF(
    df,
    equality_constraints=['var1_? + var2_? - sum_?'],
    parallelize=False,
)
df2 = m.fit()

# plots results
fig,axes = plt.subplots(3,1,sharey=True, figsize=(9,9))

df2['var2'].plot(ax=axes[0], label='forecasted var2')
df_true['var2'].plot(ax=axes[0], label='true var2')
axes[0].legend()

df2['sum'].plot(ax=axes[1], label='forecasted sum')
df_true['sum'].plot(ax=axes[1], label='true sum')
axes[1].legend()

(df2['var1'] + df2['var2'] - df2['sum']).plot(ax=axes[2], label='summation error')
axes[2].legend()
```

# Forecaster training-data retention

Package-default forecasters explicitly use `remember_data=True` on sktime
versions that support it, preserving training-data retention and update/refit
behavior for returned fitted models. Supplied forecasters keep their own
`remember_data` setting, and the original estimator is not modified. The package
locally silences only sktime's `remember_data` default-change warning while
constructing and fitting its models, including nested forecasters.

If the warning appears when constructing your own forecaster before passing it
to the package, it originates outside the package's scope. Setting
`forecaster.set_config(remember_data=False)` opts out of retention, but can change
subsequent update/refit behavior; it also cannot undo a constructor warning that
has already been emitted.

# Reproducing the paper figures

The [self-contained GDP example and figure reproduction instructions](https://github.com/sakaiando/macroframe-forecast/blob/joss_branch/paper/README.md)
are maintained on the JOSS submission branch.

# Support and contributions

For usage questions, support requests, reproducible bug reports, or feature requests, please open an issue in the
[GitHub issue tracker](https://github.com/sakaiando/macroframe-forecast/issues). See
[CONTRIBUTING.md](CONTRIBUTING.md) for what to include in a report, how to set up a development environment, run tests,
and submit a contribution.

# Disclaimer

Reuse of this tool and IMF information does not imply any endorsement of the research and/or product. Any research presented should not be reported as representing the views of the IMF, its Executive Board, or member governments.
