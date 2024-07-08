import pandas as pd
import sys
sys.path.append(r'C:\Users\dbilgin\OneDrive - International Monetary Fund (PRD)\prototype\mff')

from mff.mff.step2 import process_raw_constraints
from mff.mff.unconstrained_forecast import unconstrained_forecast

forecast_start = 2023
lam = 100

data = pd.read_excel(r'./data/input.xlsx', index_col=0, header=list(range(0, 3)), sheet_name='data').T
constraints_raw = pd.read_excel(r'./data/input.xlsx', sheet_name='constraints', header=None, index_col=0)

data.columns.name = 'variable'
data = data.stack()

C = process_raw_constraints(constraints_raw, index_iloc=range(0, 4))



unconstrained_forecast(data, forecast_start)