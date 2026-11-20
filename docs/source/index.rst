.. module:: macroframe_forecast

.. Systematic Macroframework Forecasting documentation master file, created by
   sphinx-quickstart on Mon Feb 12 12:29:05 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentation for the Macroframework Forecasting package
========================================================

Statement of need
-----------------

This package assists forecasting with constraints.

In forecasting economic time series, statistical models often need to be supplemented with procedures that impose constraints while preserving smoothness over time. Systematically imposing constraints while retaining smoothness is important but challenging. Existing packages in R and Python assist forecast reconciliation and smoothing separately but not jointly. To our knowledge, macroframe-forecast is the only package that supports the simultaneous application of both reconciliation and smoothing.

The targeted audience is policy makers that conduct multivariate economic forecasting with large *N* and small *T* (many time series and short sample periods), but the methodology can also be applied to other problems.

Installation
------------

Python 3.11 and 3.12 are the supported, tested versions. Python 3.13 and later
are outside the supported range; see
`PR #80 <https://github.com/sakaiando/macroframe-forecast/pull/80>`_
for the corresponding runtime compatibility and package metadata changes.

To install the ``macroframe-forecast`` package, run the following in the terminal/shell:

.. code-block:: console

   pip install macroframe-forecast



Quick start
-----------

The relevant import from `macroframe-foreacst` is `MFF`:

.. code-block:: python

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
   
   # input dataframe, 
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


References
----------
`A Python Package to Assist Macroframework Forecasting: Concepts and Examples (2025) <https://www.imf.org/en/Publications/WP/Issues/2025/08/29/A-Python-Package-to-Assist-Macroframework-Forecasting-Concepts-and-Examples-570041>`_

`Smooth Forecast Reconciliation (2024) <https://www.imf.org/en/Publications/WP/Issues/2024/03/22/Smooth-Forecast-Reconciliation-546654>`_

`Systematizing Macroframework Forecasting: High-Dimensional Conditional Forecasting with Accounting Identities (2023) <https://link.springer.com/article/10.1057/s41308-023-00225-8>`_


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   examples
   api_reference/index
   contributing

* :ref:`genindex`
