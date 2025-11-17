# `macroframe-forecast`: a Python package to assist with macroframework forecasting


[![!pypi](https://img.shields.io/pypi/v/macroframe-forecast?color=green)](https://pypi.org/project/macroframe-forecast/) [![Downloads](https://static.pepy.tech/personalized-badge/macroframe-forecast?period=total&units=international_system&left_color=grey&right_color=blue&left_text=cumulative%20(pypi))](https://pepy.tech/project/macroframe-forecast)

This package is based on the following papers:
* [A Python Package to Assist Macroframework Forecasting: Concepts and Examples (2025)](https://www.imf.org/en/Publications/WP/Issues/2025/08/29/A-Python-Package-to-Assist-Macroframework-Forecasting-Concepts-and-Examples-570041).
* [Smooth Forecast Reconciliation (2024)](https://www.imf.org/en/Publications/WP/Issues/2024/03/22/Smooth-Forecast-Reconciliation-546654)
* [Systematizing Macroframework Forecasting: High-Dimensional Conditional Forecasting with Accounting Identities (2023)](https://link.springer.com/article/10.1057/s41308-023-00225-8)

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
import matplotlib.pyplot as plt
from macroframe_forecast import MFF

# true data
df_true = pd.DataFrame({
    'var1': np.random.randn(30),  # 100 random values from normal distribution
    'var2': np.random.randn(30)
})
df_true['sum'] = df_true['var1'] + df_true['var2']

# input dataframe
df = df_true.copy()
fh = 5
df.iloc[-fh:, 1:] = np.nan

# apply MFF
m = MFF(df, equality_constraints=['var1_? + var2_? - sum_?'])
df2 = m.fit()

# plots results
fig,axes = plt.subplots(3,1,sharey=True, figsize=(9,9))

axes[0].plot(df2['var2'], label='forecasted var2')
axes[0].plot(df_true['var2'], label='true var2')
axes[0].legend()

axes[1].plot(df2['sum'], label='forecasted sum')
axes[1].plot(df_true['sum'], label='true sum')
axes[1].legend()

axes[2].plot( df2['var1'] + df2['var2'] - df2['sum'], label='summation error')
axes[2].legend()
```

# Disclaimer

Reuse of this tool and IMF information does not imply any endorsement of the research and/or product. Any research presented should not be reported as representing the views of the IMF, its Executive Board, or member governments.
