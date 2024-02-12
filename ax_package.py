# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 08:48:44 2022

@authors: Ando, Xiao
"""
#%% Import packages
import numpy as np
import pandas as pd
import sys

from numpy.linalg import inv
from numpy.linalg import matrix_rank
from scipy.stats import loguniform
from scipy.stats import randint
from scipy.stats import uniform
from sklearn.covariance import OAS
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNetCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

#%% main function
def ax_forecast(df, lag, Tin, C_dict, d_dict):
    """
    Parameters
    ----------
    df: dataframe
        (T+h) x m dataframe representing input data
        the first m-k columns of T:T+h rows are nan, and the rest are not nan
        if the columns are not sorted in this order, a sorted version df0 will
        be produced
    lag: int
        the number of lags used as regressors in the step1 training
        if all variables are unknown, lag should be > 0
    Tin: int
        the number of time periods in historical data used to estimate forecast-error
    C_dict: dictionary
        T+h length dictionary, each element of which is
        (m-n) x m numpy matrix C of floag type in the constraint C x df.columns = d
        the order of columns must be the same of the columns of df
        n is the number of free variables (net contents)
    d_dict: dictionary
        T+h length dictionary of numpy array, each element of which is
        (m-n) x 1 column vector in the constraint C x df = d
       
    Returns
    -------
    df2: dataframe
        (T+h) x m dataframe with nan values filled using the two-step forecasting method
    df1: dataframe
        (T+h) x m dataframe with nan values filled using the first step of the forecasting method
    df0aug_fitted_model: dictionary
        u + 2 length dictionary, each element of which is a fit object,
	containing the estimated cofficients for an unknown variable.
	The first u keys are unknown variables, and the last two store
	coefficients from regularization method and dimension reduction method

    Example
    -------
    #%% example
    import time
    
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
    df[0].iloc[-h:] = np.nan
    df[1].iloc[-h:] = np.nan
    df['one'].iloc[-h:] = np.nan
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
    
    start = time.time()
    df2,df1,df0aug_coef = ax_forecast(df, lag, Tin, C_dict, d_dict)
    #df2,df1,df0aug_coef = ax_forecast(df, lag, Tin, C_dict, d_dict)
    end = time.time()
    print('time',end-start)
    
    # forecast of the variable '0'
    pd.DataFrame({'true': df_true[0],
                  '1st stage': df1[0],
                  '2nd stage': df2[0]}).plot()
    
    # the constraints are satisfied up to numerical error
    sum(abs(df2['sum']-df2.iloc[:,:-1].sum(axis=1)))
    """
    
    df = pd.DataFrame(df) # make sure df is dataframe
    df = df.apply(pd.to_numeric) # make sure df elements are numeric

    # check1: make sure there are something to forecast
    if df.isna().any().sum() == 0:
        sys.exit('Nothing to forecast. Extend time for some variables.')
    
    # check 2: make sure the size of contraint matrix is consistent with number of vars
    for i in C_dict:
        assert(C_dict[i].shape[1] == len(df.columns))

    # check3: check whether unknown variables come before known variables, if not, change the columns of C in C_dict
    u_var = df.columns[df.isna().any()].tolist()
    k_var = df.columns[~df.isna().any()].tolist()
    correct_order = u_var + k_var
    if sum(df.columns != correct_order):
        df0 = df[correct_order]
        for i in C_dict:
            C_dict[i] = pd.DataFrame(C_dict[i],
                                     columns = df.columns)[correct_order].values
        print('df and C are re-ordered')
    else:
        df0 = df.copy()
            
    # check4: Check rank condition of the constraint matrices and drop redundant constraints
    u = len(df.loc[:,df.isna().any()].columns) # number of unknown variables
    for i in C_dict:
        Ci = C_dict[i]
        Ui = Ci.T[:u,:] # unknown part of constraint Ci, want to drop column until full rank

        # if there are redundant columns, drop them
        Ui = np.array(Ui,dtype='float')
        if matrix_rank(Ui) < Ui.shape[1]: # if columns are not full rank
            Ui_pd = pd.DataFrame(Ui) # set up a dataframe so that column index is fixed
            for col in Ui_pd.columns: # loop to drop redundant columns
                if matrix_rank(Ui_pd.drop(col,axis=1)) == matrix_rank(Ui_pd): # if redundant
                    Ui_pd = Ui_pd.drop(col,axis=1) # drop column
            idx_to_keep = Ui_pd.columns
            C_dict[i] = Ci[idx_to_keep,:]
            d_dict[i] = np.array(d_dict[i]).reshape(-1,1)[idx_to_keep,:]
                    
        # if there are no free variables, forecast can be solved by the constraints without 1st step
        Ui_new = C_dict[i].T[:u,:]
        if Ui_new.shape[0] == Ui_new.shape[1]: # if Ui is square, Ui is invertible from the previous step
            sys.exit('Error: system exactly identified, no need to use ax')

    # check5: check C and d are consistent
    u = len(df.loc[:,df.isna().any()].columns) # number of unknown variables
    for i in C_dict:
        Ci = C_dict[i]
        di = np.array(d_dict[i]).reshape(-1,1)
    assert(Ci.shape[0] == di.shape[0])
    
    # 1st step forecast
    df1, df0aug_fitted_model = step1(df0, lag, Tin)
    
    # 2nd step reconciliation
    df2 = step2(df1, df0, Tin, C_dict, d_dict)
    
    # put back the variables in the original order
    df1 = df1[df.columns]
    df2 = df2[df.columns]
    
    return df2, df1, df0aug_fitted_model

#%% sub functions

# Augment lags
def augment_lag(df,lag):
    """
    augment_lag adds lags of df
    
    Parameters
     ----------
     df: dataframe
     lag: int
         the number of lags used as regressors in the step1 training

    Returns
     -------
     dfaug: dataframe
         If df is n x m, dfaug is (n-lag) x (m x lag)
    """
    df_list = [df]
    for Li in range(1,lag+1):
        Ld = df.shift(Li)
        Ld.columns = ['L'+str(Li)+'_'+str(vn) for vn in df.columns]
        df_list.append(Ld)
    dfaug = pd.concat(df_list,axis=1).iloc[lag:,:]
    return dfaug

def step1(df0, lag, Tin):
    """
     Parameters
     ----------
     df0: dataframe
         (T+h) x m dataframe representing input data
         the first m-k columns of T:T+h rows are nan, and the rest are not nan
     lag: int
         the number of lags used as regressors in the step1 training
     Tin: int
         the number of time periods in historical data used to estimate forecast-error
     
    Returns
     -------
     df1: dataframe
         (T+h) x m dataframe
         the last Tin+h rows of the unknown variables are forecasts
         the last h rows of the known variables are filled with the forecasts of the unknowns
     df0aug_fitted_model: dictionary
         u length of dictionary.
         A key is a variable name in df0. Each element contains the fit object that
         store estimation coefficients, etc.
    """
     
    # Augment lags
    df0aug = augment_lag(df0,lag) # more columns, fewer rows
    
    # extract information on T,h,u,k from the shape of df0
    T_aug = sum(~np.isnan(df0aug).any(axis=1)) # length of historical data
    h     = len(df0aug) - T_aug # length of forecast horizon
    m_aug = df0aug.shape[1] # number of all variables
    k_aug = sum(~np.isnan(df0aug.iloc[:T_aug+1,:]).any(axis=0)) # number of known variables in T+1 including lags
    u     = m_aug - k_aug # m-k = m_aug - k_aug # number of unknown variables
    
    # create sub-dataframe and their np versions
    df0aug_u = df0aug.iloc[:,:u] # not df0_u since rows are different from df0
    df0aug_k = df0aug.iloc[:,u:]
    df0aug_u_np = df0aug_u.to_numpy()
    df0aug_k_np = df0aug_k.to_numpy()
    
    # Step1 Prediction for T+1
    df0aug_h = df0aug.copy() # hat, will be reshaped to df1 later
    df0aug_fitted_model = {} # storage for fitted model
    df0aug_h_regularization = df0aug.copy() # regularization = Elastic Net CV
    df0aug_h_dim_reduction  = df0aug.copy() # dimension reduction = Principal Component
    df0aug_h_naive          = df0aug.copy() # regularization = Elastic Net CV
    df0aug_h_kernel_ridge   = df0aug.copy() # kernel ridge
    df0aug_h_svr            = df0aug.copy() # support vector regression
    df0aug_fitted_model_regularization = pd.DataFrame(index = df0aug_u.index,
                                                      columns = df0aug.columns) # storage for ElasticNet fit
    df0aug_fitted_model_dim_reduction  = pd.DataFrame(index = df0aug_u.index,
                                                      columns = df0aug.columns) # storage for ols + pca fit
    df0aug_fitted_model_kernel_ridge   = pd.DataFrame(index = df0aug_u.index,
                                                      columns = df0aug.columns) # storage for ols + pca fit 
    df0aug_fitted_model_svr             = pd.DataFrame(index = df0aug_u.index,
                                                      columns = df0aug.columns) # storage for ols + pca fit 
    for t in range(T_aug-Tin,T_aug+1): # forecast of T-Tin to T is for the weight matrix in the 2nd step
        
        # standardize X data to be mean 0 & variance 1
        Xscaler = StandardScaler().fit(df0aug_k_np[:t,:])
        X_data = Xscaler.transform(df0aug_k_np[:t,:])
        X_pred = Xscaler.transform(df0aug_k_np[t,:].reshape(1,-1))
        
        # reduce dimension of X_data for PCA
        fit_pca = PCA(n_components = 0.9).fit(X_data) # choose #components that generates 95% variance
        X_data_reduced = fit_pca.transform(X_data)
        X_pred_reduced = fit_pca.transform(X_pred)
        
        for ui in list(range(u)):
            print('forecasting time',df0aug.index[t],'and variable',df0.columns[ui])
            
            # standardize y data to be mean 0 & variance 1
            yscaler = StandardScaler().fit(df0aug_u_np[:t,ui].reshape(-1,1)) # ensure data is 2D array
            y_data  = yscaler.transform(df0aug_u_np[:t,ui].reshape(-1,1)).ravel() # transform requires 2D array, but later fits requires 1D array
            y_mean  = yscaler.mean_
            y_std   = yscaler.scale_
            
            # when y_data is a constant, elastic net is slow, so use the mean for forecast
            if np.isclose(np.std(y_data),0):
                df0aug_fitted_model[df0.columns[ui]] = 'constant' # store model
                df0aug_h.iloc[t,ui] = y_mean
            else:
                df0aug_fitted_model[df0.columns[ui]] = 'tbd' # model to be determined by cross-validation

                # Elastic Net CV
                tscv = TimeSeriesSplit(n_splits=Tin).split(X_data) # for some reason, this cannot be placed outside ui loop
                fit_elasticnet = ElasticNetCV(cv=tscv,
                                              max_iter=100000,
                                              fit_intercept = False,
                                              n_jobs=-1).fit(X_data, y_data)
                df0aug_h_regularization.iloc[t,ui] = fit_elasticnet.predict(X_pred)[0]*y_std + y_mean # store forecast
                fit_elasticnet.Xscaler = Xscaler # store mean,std of X to forecat h+1 and beyond
                fit_elasticnet.yscaler = yscaler # store mean,std of y to forecat h+1 and beyond
                fit_elasticnet.varname = df0aug_k.columns
                df0aug_fitted_model_regularization.iloc[t,ui] = fit_elasticnet

                # OLS with X reduced by PCA
                fit_olspca = LinearRegression(fit_intercept = False).fit(X_data_reduced, y_data)
                df0aug_h_dim_reduction.iloc[t,ui] = fit_olspca.predict(X_pred_reduced)[0]*y_std + y_mean # store forecast
                fit_olspca.Xscaler = Xscaler # store X's reduced space to forecat h+1 and beyond
                fit_olspca.yscaler = yscaler # store mean,std of y to forecat h+1 and beyond
                fit_olspca.fit_pca = fit_pca # store X's reduced space to forecat h+1 and beyond
                df0aug_fitted_model_dim_reduction.iloc[t,ui] = fit_olspca
                
                # naive estimator
                df0aug_h_naive.iloc[t,ui] = df0aug_h_naive.iloc[t-1,ui]

                # Kernel ridge
                param_distributions = {
                    'alpha': loguniform(0.1, 1000),
                    'gamma': uniform(0.5*1/X_data.shape[1] , 2*1/X_data.shape[1])
                }
                tscv = TimeSeriesSplit(n_splits=Tin).split(X_data) # for some reason, this cannot be placed outside ui loop
                kernel_ridge_tuned = RandomizedSearchCV(
                    KernelRidge(kernel='rbf'),
                    param_distributions = param_distributions,
                    n_iter = 500,
                    random_state = 0,
                    cv = tscv,
                    n_jobs = -1
                )
                fit_kernel_ridge = kernel_ridge_tuned.fit(X_data, y_data)
                df0aug_h_kernel_ridge.iloc[t,ui] = fit_kernel_ridge.predict(X_pred)[0]*y_std + y_mean # store forecast
                fit_kernel_ridge.Xscaler = Xscaler # store X's reduced space to forecat h+1 and beyond
                fit_kernel_ridge.yscaler = yscaler # store mean,std of y to forecat h+1 and beyond
                fit_kernel_ridge.fit_kernel_ridge = fit_kernel_ridge # store X's reduced space to forecat h+1 and beyond
                df0aug_fitted_model_kernel_ridge.iloc[t,ui] = fit_kernel_ridge
                
                # Support vector regression
                param_distributions = {
                    'C': loguniform(0.1, 1000)
                }
                tscv = TimeSeriesSplit(n_splits=Tin).split(X_data) # for some reason, this cannot be placed outside ui loop
                svr_tuned = RandomizedSearchCV(
                    SVR(kernel='rbf'),
                    param_distributions = param_distributions,
                    n_iter = 500,
                    random_state = 0,
                    cv = tscv,
                    n_jobs = -1
                )
                fit_svr = svr_tuned.fit(X_data, y_data)
                df0aug_h_svr.iloc[t,ui] = fit_svr.predict(X_pred)[0]*y_std + y_mean # store forecast
                fit_svr.Xscaler = Xscaler # store X's reduced space to forecat h+1 and beyond
                fit_svr.yscaler = yscaler # store mean,std of y to forecat h+1 and beyond
                fit_svr.fit_svr = fit_svr # store X's reduced space to forecat h+1 and beyond
                df0aug_fitted_model_svr.iloc[t,ui] = fit_svr
                
    # Using cross-validation to choose fromm {ElasticNetCV,OLS+PCA} that performs best for each unknown variable
    for ui in list(range(u)):
        # use cross-validation only if y variable has variation
        if df0aug_fitted_model[df0.columns[ui]] != 'constant':
            # compare forecast error from regularization and reduction
            fe_regularization = np.absolute( (df0aug_h_regularization.iloc[T_aug-Tin:T_aug,ui] -
                                                               df0aug.iloc[T_aug-Tin:T_aug,ui]) ).mean()
            fe_dim_reduction  = np.absolute(  (df0aug_h_dim_reduction.iloc[T_aug-Tin:T_aug,ui] -
                                                               df0aug.iloc[T_aug-Tin:T_aug,ui]) ).mean()
            fe_naive          = np.absolute(  (df0aug_h_naive.iloc[T_aug-Tin:T_aug,ui] -
                                                               df0aug.iloc[T_aug-Tin:T_aug,ui]) ).mean()
            fe_kernel_ridge   = np.absolute(  (df0aug_h_kernel_ridge.iloc[T_aug-Tin:T_aug,ui] -
                                                               df0aug.iloc[T_aug-Tin:T_aug,ui]) ).mean()
            fe_svr            = np.absolute(  (df0aug_h_svr.iloc[T_aug-Tin:T_aug,ui] -
                                                               df0aug.iloc[T_aug-Tin:T_aug,ui]) ).mean()
            fe_min = min(fe_regularization,
                         fe_dim_reduction,
                         fe_naive,
                         fe_kernel_ridge,
                         fe_svr)
            if fe_regularization == fe_min:
                print(df0.columns[ui],'regularization')
                df0aug_h.iloc[:,ui] = df0aug_h_regularization.iloc[:,ui]
                df0aug_fitted_model[df0.columns[ui]] = df0aug_fitted_model_regularization.iloc[T_aug,ui]
            elif fe_dim_reduction == fe_min:
                print(df0.columns[ui],'dimension reduction')
                df0aug_h.iloc[:,ui] = df0aug_h_dim_reduction.iloc[:,ui]
                df0aug_fitted_model[df0.columns[ui]] = df0aug_fitted_model_dim_reduction.iloc[T_aug,ui]
            elif fe_naive == fe_min:
                print(df0.columns[ui],'naive random walk')
                df0aug_fitted_model[df0.columns[ui]] = 'naive'
                df0aug_h.iloc[:,ui] = df0aug_h_naive.iloc[:,ui]
            elif fe_kernel_ridge == fe_min:
                print(df0.columns[ui],'kernel ridge')
                df0aug_h.iloc[:,ui] = df0aug_h_kernel_ridge.iloc[:,ui]
                df0aug_fitted_model[df0.columns[ui]] = df0aug_fitted_model_kernel_ridge.iloc[T_aug,ui]
            elif fe_svr == fe_min:
                print(df0.columns[ui],'svr')
                df0aug_h.iloc[:,ui] = df0aug_h_svr.iloc[:,ui]
                df0aug_fitted_model[df0.columns[ui]] = df0aug_fitted_model_svr.iloc[T_aug,ui]
    
    # store unused fitted models
    df0aug_fitted_model['regularization'] = df0aug_fitted_model_regularization
    df0aug_fitted_model['dim_reduction']  = df0aug_fitted_model_dim_reduction
    df0aug_fitted_model['kernel_ridge']  = df0aug_fitted_model_kernel_ridge
    df0aug_fitted_model['svr']  = df0aug_fitted_model_svr

    # forecast of T+2 to T+h, if h = 1 nothing will happen
    for t in range(-h+1, 0):

        # drop lag variables and re-augment
        df0_h = df0.copy()
        df0_h.iloc[-h-Tin:,:u] = df0aug_h.iloc[-h-Tin:,:u]
        df0aug_h = augment_lag(df0_h,lag)
        
        for ui in range(u):
            if df0aug_fitted_model[df0.columns[ui]] == 'constant':
               df0aug_h.iloc[t,ui] = df0aug_h.iloc[-h,ui] # use T+1 if u_i is a constant
            elif df0aug_fitted_model[df0.columns[ui]] == 'naive':
                df0aug_h.iloc[t,ui] = df0aug_h.iloc[t-1,ui]
            else:
                model = df0aug_fitted_model[df0.columns[ui]]
                X_pred = model.Xscaler.transform(df0aug_h.iloc[t,u:].values.reshape(1,-1)) # extract X and transform
                y_mean = model.yscaler.mean_
                y_std  = model.yscaler.scale_
                if type(model) == LinearRegression:
                    X_pred_reduced = model.fit_pca.transform(X_pred)
                    df0aug_h.iloc[t,ui] = model.predict(X_pred_reduced)[0]*y_std + y_mean
                else:
                    df0aug_h.iloc[t,ui] = model.predict(X_pred)[0]*y_std + y_mean

    # drop lags and add the rows that were dropped when lags are augmented
    df1 = df0.copy()
    df1.iloc[-h-Tin:,:u] = df0aug_h.iloc[-h-Tin:,:u]
    
    # reorder variables to match df0
    df1 = df1[df0.columns]
    
    return df1, df0aug_fitted_model

def step2(df1, df0, Tin, C_dict, d_dict):
    """
     Parameters
     ----------
     df0: dataframe
         (T+h) x m dataframe representing input data
         the first m-k columns of T:T+h rows are nan, and the rest are not nan
     df1: dataframe
         (T+h) x m dataframe representing 1st stage forecast
         no columns contain nan
     Tin: int
         the number of time periods in historical data used to estimate forecast-error
     C_dict: dictionary
         T+h length dictionary, each element of which is
         (m-n) x m numpy matrix C in the constraint C x df = d
         the order of columns must be the same of the columns of df
         n is the number of free variables (net contents)
     d_dict: dictionary
         T+h length dictionary of numpy array, each element of which is
         (m-n) x 1 column vector in the constraint C x df = d
    
    Returns
     -------
     df2: dataframe
         (T+h) x m dataframe
         the last h rows are forecast that satisfies C x df=d
    """
    # extract information on T,h,u, from df0
    T = sum(~np.isnan(df0).any(axis=1))
    h = len(df0)-T
    df0_u = df0.loc[:,df0.isna().any()]
    u = len(df0_u.columns)
    df0_u = df0_u.to_numpy()
    
    df1_u = df1.iloc[:,:u].to_numpy()
    df1_k = df1.iloc[:,u:].to_numpy()
    
    # construct weight matrix
    eh = df1_u[T-Tin:T,:] - df0_u[T-Tin:T,:] # in-sample one-step ahead forecast error
    W  = OAS().fit(eh).covariance_

    # reconcili rh by projecting it on constraint
    df2_u = df1_u.copy()
    for hi in range(h):
        C = np.array(C_dict[T+hi],dtype='float') # to avoid error in inv(U.T @ W @ U)
        U = C.T[:u,:]
        d = d_dict[T+hi]
	# this step may need python 3.8 or above, 3.6 may gives an error
        df2_u[T+hi:T+hi+1,:] = ( df1_u[T+hi:T+hi+1,:].T - W @ U @ inv(U.T @ W @ U) @ \
             (C @ np.concatenate( (df1_u[T+hi:T+hi+1,:],df1_k[T+hi:T+hi+1,:]), axis=1 ).T -d)).T

    df2 = np.concatenate([df2_u,df1_k],axis=1)
    df2 = pd.DataFrame(df2, index  = df1.index,
                            columns= df1.columns)
    df2.iloc[:T,:] = df0.iloc[:T,:]
    
    return df2
