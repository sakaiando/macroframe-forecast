# `macroframe-forecast`: a Python package to assist with macroframework forecasting


[![!pypi](https://img.shields.io/pypi/v/macroframe-forecast?color=green)](https://pypi.org/project/macroframe-forecast/) [![Downloads](https://static.pepy.tech/personalized-badge/macroframe-forecast?period=total&units=international_system&left_color=grey&right_color=blue&left_text=cumulative%20(pypi))](https://pepy.tech/project/macroframe-forecast)

This repository contains the Python code to assist macroframework forecasting. The methodologies are described in [Systematizing Macroframework Forecasting: High-Dimensional Conditional Forecasting with Accounting Identities](https://link.springer.com/article/10.1057/s41308-023-00225-8) and [Smooth Forecast Reconciliation](https://www.imf.org/en/Publications/WP/Issues/2024/03/22/Smooth-Forecast-Reconciliation-546654).

# Documentation

Please refer to [this link](https://sakaiando.github.io/macroframe-forecast/) for documentation.

# Installation

To install the `macroframe-forecast` package, run the following from the repository root:

```shell
pip install macroframe-forecast
```

# Quick start

```python
import numpy as np
import pandas as pd
from sktime.datasets import load_macroeconomic

from macroframe_forecast import MFF

df_true = load_macroeconomic().iloc[:, :5]

# input dataframe
df = df_true.copy()
fh = 5
df.iloc[-fh:, 0] = np.nan

# apply MFF
m = MFF(df, equality_constraints=[])
df2 = m.fit()
df0 = m.df0
df1 = m.df1
df1_model = m.df1_model
smoothness = m.smoothness
shrinkage = m.shrinkage

# plot results
t0 = -30
ax = df0.iloc[t0:, 0].plot(label="df0")
df1.iloc[t0:, 0].plot(ax=ax, label="df1")
df2.iloc[t0:, 0].plot(ax=ax, label="df2")
df_true.iloc[t0:, 0].plot(ax=ax, label="df_true")
ax.axvline(x=df0.index[-fh])
ax.legend()

print("smoothness", smoothness.values)
print("shrinkage", np.round(shrinkage, 3))
for ri, ci in np.argwhere(df.isna()):
    print(df1_model.index[ri], df1_model.columns[ci], df1_model.iloc[ri, ci].best_params_)
```
