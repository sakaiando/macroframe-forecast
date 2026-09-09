.. module:: mff

.. Systematic Macroframework Forecasting documentation master file, created by
   sphinx-quickstart on Mon Feb 12 12:29:05 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentation for the Macroframework Forecasting package.
====================================================================

This repository contains the Python code for the forecasting method described in:

`A Python Package to Assist Macroframework Forecasting: Concepts and Examples (2025) <https://www.imf.org/en/Publications/WP/Issues/2025/08/29/A-Python-Package-to-Assist-Macroframework-Forecasting-Concepts-and-Examples-570041>`_.

`Smooth Forecast Reconciliation (2024) <https://www.imf.org/en/Publications/WP/Issues/2024/03/22/Smooth-Forecast-Reconciliation-546654>`_.

`Systematizing Macroframework Forecasting: High-Dimensional Conditional Forecasting with Accounting Identities (2023) <https://link.springer.com/article/10.1057/s41308-023-00225-8>`_.

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
   import matplotlib.pyplot as plt
   from macroframe_forecast import MFF
   
   # true data
   df_true = pd.DataFrame({
       'var1': np.random.randn(30),  # 100 random values from normal distribution
       'var2': np.random.randn(30)
   })
   df_true['sum'] = df_true['var1'] + df_true['var2']
   
   # input dataframe, 
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


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   examples
   mff_documentation
   contributing

* :ref:`genindex`
