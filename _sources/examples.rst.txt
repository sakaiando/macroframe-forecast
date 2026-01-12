Examples
--------

Single-variable example
~~~~~~~~~~~~~~~~~~~~~~~


.. code-block:: python

  import pandas as pd
  import numpy as np
  from macroframe_forecast import MFF
  import macroframe_forecast
  from string import ascii_uppercase, ascii_lowercase
  from sktime.datasets import load_macroeconomic
  import matplotlib.pyplot as plt
  
  #%% Reading the data and generating forecasts.
  
  # Reading GDP data as a pandas dataframe.
  # This dataframe has two columns: year and GDP. Data from 2024-2029 are WEO forecasts. 
  from pandas import DataFrame
  
  GDP_data_true = DataFrame({
      "year": [
          1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959,
          1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969,
          1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979,
          1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989,
          1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999,
          2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
          2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
          2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029,
          2030
      ],
      "GDP": [
          301782704906.154, 348993057004.926, 368027835977.609, 389147698401.843,
          390276672099.46, 424868331217.657, 448388356231.708, 471707274214.225,
          478166880805.205, 519476064642.104, 539899866168.654, 558583293630.287,
          600454646133.34, 633368190949.311, 680153540812.135, 737201978910.734,
          808045440847.441, 853883822469.0601, 933096436159.1281, 1008751520510.61,
          1064366709379.28, 1155403629216.3, 1269884411457.22, 1418456050381.57,
          1536647924378.57, 1674009506825.93, 1867242215504.46, 2079644632633.34,
          2350400768409.49, 2627325000000.0, 2857325000000.0, 3207025000000.0,
          3343800000000.0, 3634025000000.0, 4037650000000.0, 4339000000000.0,
          4579625000000.0, 4855250000000.0, 5236425000000.0, 5641600000000.0,
          5963125000000.0, 6158125000000.0, 6520325000000.0, 6858550000000.0,
          7287250000000.0, 7639750000000.0, 8073125000000.0, 8577550000000.0,
          9062825000000.0, 9631175000000.0, 10250950000000.0, 10581925000000.0,
          10929100000000.0, 11456450000000.0, 12217175000000.0, 13039200000000.0,
          13815600000000.0, 14474250000000.0, 14769850000000.0, 14478050000000.0,
          15048975000000.0, 15599725000000.0, 16253950000000.0, 16880675000000.0,
          17608125000000.0, 18295000000000.0, 18804900000000.0, 19612100000000.0,
          20656525000000.0, 21539975000000.0, 21354125000000.0, 23681175000000.0,
          26006900000000.0, 27720725000000.0, 29184900000000.0, 30507217002511.25,
          31717641479090.75, 32941710359665.25, 34342131994149.0, 35712823521822.0,
          37153089058192.75
      ]
  })
  
  
  # Forecasted GDP growth in 2029 (last year) is as given below
  final_year_growth =  100*(GDP_data_true.iloc[-1,1]/GDP_data_true.iloc[-2,1]-1)
  
  # The original GDP data is in dollar numbers, but changing this to billions 
  # going forward in order to deal with problem of matrix invertibility.
  GDP_data_true['GDP'] = GDP_data_true['GDP']/1e12
  
  # Time period hs to be set as the index. Here year is the time identifier, 
  # therefore setting this as the index. 
  GDP_data_true.set_index(GDP_data_true['year'], inplace = True)
  GDP_data_true.drop(columns = 'year', inplace = True)
  
  # Creating a copy which is used for geenrating the forecasts. Removing the last
  # six years of data for ease of forecasts
  GDP_data = GDP_data_true.copy()
  # Removing the last six years of data so that they are forecasted by the 
  # function. 
  GDP_data.iloc[-6:,0] = np.nan
  
  # Now we assume that US GDP grows by 4% from 2028 to 2029, which is given by the 
  # WEO forecast. This therefore works as a constraint for the forecasts.
  # The dataframe has GDP in levels terms, therefore the constraint has to be 
  # specified in levels terms as well. The constraints can be rewritten in the following
  # steps.
  # GDP_2029/GDP_2028 - 1 = 0.04
  # GDP_2029 = 1.04*GDP_2028 
  # GDP_2029 - 1.04*GDP_2028 = 0
  
  # Constraints are to be provided in the form of a list, even when there is only 
  # constraint.
  GDP_constraint = ['GDP_2030 - 1.04*GDP_2029']
  
  m = MFF(df = GDP_data,
          equality_constraints = GDP_constraint,
          parallelize = False)
  
  # Using the fit method generates first as well as second step forecasts.
  m.fit()
  
  # First step forecasts are stored as df1 in the fitted object.
  firststep_GDP = m.df1
  
  # The forecasted data is filled into the df2 dataframe in the fitted object.
  reconciled_GDP = m.df2
  
  # Models are stored in a dataframe in the fitted object.
  
  models_used = m.df1_model
  models_used.iloc[-1,0]
  #%% Plotting first and second step forecasts 
  fig, ax = plt.subplots(figsize=(8, 4.8)) 
  
  firststep_GDP['GDP'].plot(ax=ax, label='First step forecasts', linestyle = '--')
  reconciled_GDP['GDP'].plot(ax=ax, label='Final forecasts', linestyle = '-.')
  GDP_data['GDP'].plot(ax = ax, label = 'Known values', color = 'red')
  
  ax.set_xlabel('Year')  
  ax.set_ylabel('US Nominal GDP (in US$ trn)')  
  ax.set_title('US GDP in levels')  
  ax.legend(loc = 'lower left')
  
  # max_xlastvalue = reconciled_GDP.index.max()
  ax.set_xlim([2020, 2030])
  ax.set_ylim([15, 40])
  
  plt.xticks(np.arange(2019, 2031,2))
  
  plt.show()
  
  # %%
  
  firststep_GDP['GDP_growth'] = (firststep_GDP['GDP']/firststep_GDP['GDP'].shift(1) - 1)*100
  reconciled_GDP['GDP_growth'] = (reconciled_GDP['GDP']/reconciled_GDP['GDP'].shift(1) - 1)*100
  GDP_data['GDP_growth'] = (GDP_data['GDP']/GDP_data['GDP'].shift(1) - 1)*100
  
  fig, ax = plt.subplots(figsize=(8, 4.8))
  
  firststep_GDP['GDP_growth'].plot(ax=ax, label='First-step forecasts', linestyle = '--')
  reconciled_GDP['GDP_growth'].plot(ax=ax, label='Second-step forecasts', linestyle = '-.')
  GDP_data['GDP_growth'].plot(ax = ax, label = 'Known values', color = 'red')
  
  ax.set_xlabel('Year')  
  ax.set_ylabel('Nominal GDP growth (annual, %)')  
  ax.set_title('US GDP growth rates')  
  ax.legend(loc = 'upper left')
  
  # Add triangle marker at (2029, 4)
  ax.plot(2030, 4, marker='v', color='black', markersize=8, label='_nolegend_')
  
  # Add text annotation
  ax.annotate('2030 growth constraint', xy=(2030, 4), xytext=(2030-2, 2.5),
              arrowprops=dict(arrowstyle='->', color='black'), color='black')
  
  # max_xlastvalue = reconciled_GDP.index.max()
  ax.set_xlim([2019, 2031])
  
  plt.xticks(np.arange(2020, 2031,2))
  
  plt.show()
  
  # %% Looking at externally generated first-stage
  
  GDP_forecasts_external = pd.DataFrame({"GDP": [29.0, 31.5, 33, 34.1,36.8, 39]},
                              index = [2025, 2026, 2027, 2028, 2029, 2030])
  
  # Build MultiIndex using column name
  multi_index = pd.MultiIndex.from_product([[GDP_forecasts_external.columns[0]], GDP_forecasts_external.index],
                                           names=[None, 'year'])
  
  # Correct: flatten the 2D array to 1D
  GDP_multiindex_series = pd.Series(GDP_forecasts_external.values.ravel(), index=multi_index)
  
  
  
  W_alt =  pd.DataFrame(np.eye(len(multi_index)), index=multi_index, columns=multi_index)  # Create identity matrix with shape (n x n)
  
  smoothness_alt = pd.Series(np.ones(1) * 100, index=[multi_index])
  
  Phi_alt = macroframe_forecast.utils.GenSmoothingMatrix(W_alt, smoothness_alt)
  
  
  final_forecasts = macroframe_forecast.utils.Reconciliation(y1 = GDP_multiindex_series, 
                                                             W = m.W, Phi = m.Phi, 
                                                             C = m.C, d = m.d, 
                                                             C_ineq = m.C_ineq, 
                                                             d_ineq = m.d_ineq)
  # %%
  
  # Convert MultiIndex Series to regular Series with year index
  
  gdp_to_forecast_series = GDP_data
  
  gdp_series = GDP_multiindex_series.xs('GDP', level=0)
  second_stage_series = final_forecasts.xs('GDP', level=0)
  
  # Now plot it
  fig, ax = plt.subplots(figsize=(8, 4.8))
  gdp_series.plot(ax=ax, label='Externally generated first-step forecasts', linestyle='--')
  second_stage_series.iloc[:,0].plot(ax=ax, label='Second-step forecasts', linestyle = '-.')
  
  
  
  # Add labels and formatting
  ax.set_xlabel('Year')
  ax.set_ylabel('US Nominal GDP (in US$ trn)')  
  ax.set_title('US GDP in levels') 
  ax.legend(loc='upper left')
  ax.set_xlim([2024, 2030])
  ax.set_ylim([15, 40])


Multi-variable example
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

  import pandas as pd
  import numpy as np
  from macroframe_forecast import MFF
  import matplotlib.pyplot as plt
  from sktime.forecasting.compose import DirectReductionForecaster
  from sktime.forecasting.compose import ForecastingPipeline
  from sklearn.linear_model import LinearRegression  
  from pandas import DataFrame
  
  data = DataFrame({
      "year": [
          2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
          2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020,
          2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030
      ],
      "exp": [
          32.801, 33.698, 34.037, 33.719, 33.928, 33.692, 34.562, 37.144, 41.399, 39.763,
          38.796, 37.223, 35.782, 35.324, 35.031, 35.333, 35.194, 35.349, 35.819, 44.779,
          43.218, 36.829, 37.113, 37.593, 37.848, 38.004, 38.107, 38.024, 37.711, 37.862
      ],
      "rev": [
          32.257, 29.877, 29.266, 29.476, 30.853, 31.656, 31.649, 30.532, 28.222, 28.770,
          29.080, 29.109, 31.222, 31.298, 31.501, 30.977, 30.400, 30.014, 30.014, 30.631,
          31.827, 33.130, 29.949, 30.331, 31.389, 32.514, 32.754, 32.409, 32.222, 32.248
      ],
      "int_payments": [
          3.255, 2.892, 2.658, 2.563, 2.704, 2.775, 2.933, 2.776, 2.574, 2.678,
          2.880, 2.726, 2.485, 2.474, 2.341, 2.490, 2.522, 2.769, 2.817, 2.537,
          2.669, 3.137, 3.600, 4.195, 4.301, 4.427, 4.451, 4.370, 4.353, 4.290
      ],
      "pb": [
          2.711, -0.929, -2.113, -1.681, -0.371, 0.739, 0.020, -3.836, -10.603, -8.315,
          -6.836, -5.387, -2.076, -1.552, -1.189, -1.867, -2.272, -2.566, -2.988, -11.610,
          -8.721, -0.561, -3.564, -3.067, -2.158, -1.063, -0.902, -1.246, -1.136, -1.324
      ]
  })
  
  
  # Data upto 2024 is known for all variables. 2024 onwards data are all WEO forecasts.
  # Let us assume that the path for Primary Balance/GDP is known to the forecaster,
  # which is given by the WEO forecasts, while the other three variables are to be
  # forecasted. These unknown values are therefore replaced by NaNs.
  
  fiscal_data = fiscal_data_true.copy()
  fiscal_data.iloc[-6:,:3] = np.nan
  
  # fiscal_data.iloc[-1,0] = fiscal_data_true.iloc[-1,0].copy()
  
  # The basic acccounting identiy can be writted as:
  # Primary Balance/GDP = Revenue /GDP - Expenditure/GDP + Interest Payments/GDP
  # We know that this identity has to bind throughout the forecasting horizon, and
  # therefore we can specify this using the wildcard feature.
  
  fiscal_constraint = ['pb? - rev? + exp? - int_payments?',
                       'exp_2030 - 37']
  
  # Defining the OLS forecasting pipeline for the example
  
  ols = ForecastingPipeline(steps=[
      ('ols',DirectReductionForecaster(LinearRegression()))
      ])
  
  m = MFF(df = fiscal_data,
          equality_constraints = fiscal_constraint,
          forecaster = ols,
          parallelize = False) 
  
  m.fit()
  
  first_step_forecasts = m.df1
  second_step_forecasts = m.df2
  
  # final_forecasts = 
  # %% Expenditure forecasts
  fig, ax = plt.subplots(figsize=(8, 4.8)) 
  
  first_step_forecasts['exp'].plot(ax=ax, label='First-step forecasts', linestyle = '--')
  second_step_forecasts['exp'].plot(ax=ax, label='Second-step forecast', linestyle = '-.')
  fiscal_data['exp'].plot(ax = ax, label = 'WEO values', color = 'red')
  
  ax.set_xlabel('Year')  
  ax.set_ylabel('US Government Expenditure to GDP ratio (%)')  
  ax.set_title('Government Expenditure')  
  ax.legend(loc = 'lower left')
  
  ax.plot(2030, 36.65, marker='v', color='black', markersize=8, label='_nolegend_')
  
  # Add text annotation
  ax.annotate('2030 expenditure constraint value', xy=(2030, fiscal_data_true.iloc[-1,0]), xytext=(2027, 40),
              arrowprops=dict(arrowstyle='->', color='black'), color='black')
  
  # max_xlastvalue = reconciled_GDP.index.max()
  ax.set_xlim([2021, 2030])
  
  plt.xticks(np.arange(2021, 2031,2))
  
  plt.show()
  # %% Revenue forecasts
  fig, ax = plt.subplots(figsize=(8, 4.8)) 
  
  first_step_forecasts['rev'].plot(ax=ax, label='First-step forecast', linestyle = '--')
  second_step_forecasts['rev'].plot(ax=ax, label='Second-step forecast', linestyle = '-.')
  fiscal_data['rev'].plot(ax = ax, label = 'WEO values', color = 'red')
  
  ax.set_xlabel('Year')  
  ax.set_ylabel('US Government Revenue to GDP ratio (%)')  
  ax.set_title('Government Revenue')  
  ax.legend(loc = 'lower left')
  
  # max_xlastvalue = reconciled_GDP.index.max()
  ax.set_xlim([2021, 2030])
  
  plt.xticks(np.arange(2021, 2031,2))
  
  plt.show()
  
  # %% Interest Payment forecasts
  fig, ax = plt.subplots(figsize=(8, 4.8)) 
  
  first_step_forecasts['int_payments'].plot(ax=ax, label='First-step forecast', linestyle = '--')
  second_step_forecasts['int_payments'].plot(ax=ax, label='Second-step forecast', linestyle = '-.')
  fiscal_data['int_payments'].plot(ax = ax, label = 'WEO values', color = 'red')
  
  ax.set_xlabel('Year')  
  ax.set_ylabel('US Government Interest Payments to GDP ratio (%)')  
  ax.set_title('Interest Payments')  
  ax.legend(loc = 'lower left')
  
  # max_xlastvalue = reconciled_GDP.index.max()
  ax.set_xlim([2021, 2030])
  
  plt.xticks(np.arange(2021, 2031,2))
  
  plt.show()
  
  
  
  # %% First step primary balance vs. the constraints
  
  first_step_forecasts['pb_calculated'] = first_step_forecasts['rev'] - first_step_forecasts['exp'] + first_step_forecasts['int_payments']
  second_step_forecasts['pb_calculated'] = second_step_forecasts['rev'] - second_step_forecasts['exp'] + second_step_forecasts['int_payments']
  
  fig, ax = plt.subplots(figsize=(8, 4.8)) 
  
  first_step_forecasts['pb_calculated'].plot(ax=ax, label='First-step forecast', linestyle = '--')
  second_step_forecasts['pb_calculated'].plot(ax=ax, label='Second-step forecast', linestyle = '-.')
  
  # fiscal_data[fiscal_data]['pb'].plot(ax=ax, label='WEO values', linestyle = '-.')
  fiscal_data[fiscal_data.index<2024]['pb'].plot(ax = ax, label = 'WEO values', color = 'red')
  fiscal_data[fiscal_data.index>=2024]['pb'].plot(ax = ax, label = 'Constraint values', color = 'green', marker = 'o', linestyle = 'None')
  
  
  ax.set_xlabel('Year')  
  ax.set_ylabel('Primary Balance to GDP ratio (%)')  
  ax.set_title('Primary Balance')  
  ax.legend(loc = 'lower left')
  
  # max_xlastvalue = reconciled_GDP.index.max()
  ax.set_xlim([2021, 2030])
  
  plt.xticks(np.arange(2021, 2031,2))
  
  plt.show()
  # %%
  
