# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 08:48:44 2022

@authors: Ando, Xiao
"""
from ax_package import ax_forecast
import pandas as pd
import numpy as np

#%% main function
def test_ax_forecast():
    np.random.seed(0)
    T = 30
    h = 3
    num_free_vars = 40
    df_true = pd.DataFrame(np.random.normal(0,1,[T+h,num_free_vars]))
    df_true['one'] = 1 # constant
    df_true['sum'] = df_true.iloc[:,:].sum(axis=1)
    
    # constraint in true data
    df_true['sum']-df_true.iloc[:,:-1].sum(axis=1)
    
    num_variables = num_free_vars + 2
    
    df = df_true.copy()
    df.iloc[-h:, df.columns.get_loc(0)] = np.nan
    df.iloc[-h:, df.columns.get_loc(1)] = np.nan
    df.iloc[-h:, df.columns.get_loc('one')] = np.nan
    C = np.ones([1,num_variables])
    C[0,-1] = -1
    d = 0
    C_dict = {}
    d_dict = {}
    for i in range(T+h):
        C_dict[i] = C
        d_dict[i] = d
    
    lag = 1
    Tin = 5
    df
    
    df2,df1,df0aug_coef = ax_forecast(df, lag, Tin, C_dict, d_dict)
    df1.to_csv('example_df1_temp.csv')
    df2.to_csv('example_df2_temp.csv')
    df1 = pd.read_csv('example_df1_temp.csv')
    df2 = pd.read_csv('example_df2_temp.csv')
    
    expected_df2 = pd.read_csv('example_df2.csv')
    expected_df1 = pd.read_csv('example_df1.csv')
    
    assert df1.equals(expected_df1)
    assert df2.equals(expected_df2)