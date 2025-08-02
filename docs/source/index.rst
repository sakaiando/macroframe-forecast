.. module:: mff

.. Systematic Macroframework Forecasting documentation master file, created by
   sphinx-quickstart on Mon Feb 12 12:29:05 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentation for the Macroframework Forecasting package.
====================================================================

This repository contains the Python code for the forecasting method described in:

`Systematizing Macroframework Forecasting: High-Dimensional Conditional Forecasting with Accounting Identities <https://link.springer.com/article/10.1057/s41308-023-00225-8>`_.

`Smooth Forecast Reconciliation <https://www.imf.org/en/Publications/WP/Issues/2024/03/22/Smooth-Forecast-Reconciliation-546654>`_.

Installation
------------

To install the `macroframe-foreacst` package, run the following in the terminal/shell:

.. code-block:: console

   pip install macroframe-forecast



Quick start
-----------

The relevant import from `macroframe-foreacst` is `MFF`:

.. code-block:: python

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


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   examples
   mff_documentation
   contributing

* :ref:`genindex`
