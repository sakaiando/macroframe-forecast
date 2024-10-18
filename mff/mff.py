# Disclaimer: Reuse of this tool and IMF information does not imply
# any endorsement  of the research and/or product. Any research presented
# should not be reported as representing the views of the IMF,
# its Executive Board, member governments.

from scipy.linalg import block_diag
from dask import delayed
from numpy.linalg import inv
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.compose import ( 
    DirectReductionForecaster,
    ForecastingPipeline,
    MultiplexForecaster,
    TransformedTargetForecaster
)
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.forecasting.naive import NaiveForecaster
from sktime.split import ExpandingGreedySplitter
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from string import ascii_uppercase, ascii_lowercase
from time import time
from typing import List

import copy
import dask
import random
import re
import scipy
import cvxpy as cp
import numpy as np
import pandas as pd
import sympy as sp
import warnings

#%% MFF

def DefaultForecaster()->BaseForecaster:
    """
    

    Returns
    -------
    BaseForecaster
        DESCRIPTION.

    """
    
    pipe_y_elasticnet = TransformedTargetForecaster(
        steps=[
            ('scaler', TabularToSeriesAdaptor(StandardScaler())),
            ('forecaster',DirectReductionForecaster(ElasticNetCV(max_iter=5000))),
        ]
    )
    pipe_yX_elasticnet = ForecastingPipeline(
        steps=[
            ('scaler', TabularToSeriesAdaptor(StandardScaler())),
            ('pipe_y', pipe_y_elasticnet),
        ]
    )
    
    # forecaster representation for selection among the listed models
    forecaster = MultiplexForecaster(
        forecasters=[
            ('naive_drift', NaiveForecaster(strategy='drift',window_length=5)),
            ('naive_last', NaiveForecaster(strategy='last')),
            ('naive_mean', NaiveForecaster(strategy='mean',window_length=5)),
            ('elasticnetcv', pipe_yX_elasticnet)
        ],
    )
    
    cv = ExpandingGreedySplitter(test_size=1, folds=5)
    
    gscv = ForecastingGridSearchCV(
        forecaster=forecaster,
        cv=cv,
        param_grid={'selected_forecaster':[
                'naive_drift',
                'naive_last',
                'naive_mean',
                'elasticnetcv'
                ],},
        backend='dask'
    )

    return gscv


class MFF:
    
    """A class for Macro-Framework Forecasting (MFF). 
    
    This class facilitates forecasting of single frequency time series data 
    using a two-step process. First step of the forecasting procedure generates
    unconstrained forecasts using the forecaster specified. In the next step, 
    these forecasts are then reconclied so that they satisfy the supplied 
    constrants, and smoothness of the forecasts is maintained.

    Parameters
    ----------
    df : pd.DataFrame
       Input dataframe containing time series data. Data should be in wide
       format, with each row containing data for one period, and each 
       column containing data for one variable.
    
    forecaster : Forecaster pipeline
        Forecasting pipeline to be used for predictions. (In development, to change)
    
    constraints_with_wildcard : str, optional(default: None)
        Constraints that hold with equality. Constraints may include wildcard, 
        in which case constraints will be applied across all horizons, or
        may be defined for specified time periods.
     
    ineq_constraints_with_wildcard : str, optional(default: None)
        Inequality constraints, comparable to ``constraints_with_wildcard``.
        Constraints may include wildcard, in which case constraints will be
        applied across all horizons, or may be defined for specified time 
        periods.
       
    parrallelize : boolean
        Indicate whether paralellization should be employed for generating 
        first step forecasts  

    Returns
    -------
    df2 : pd.Dataframe
        Output dataframe with all reconciled forecasts filled into the original
        input. 
    """
    def __init__(self,
                 df: pd.DataFrame,
                 forecaster = DefaultForecaster(),
                 constraints_with_wildcard:List[str] = [],
                 ineq_constraints_with_wildcard:List[str] = [],
                 parallelize:bool = True):
        
        self.df = df
        self.forecaster = forecaster
        self.constraints_with_wildcard = constraints_with_wildcard
        self.ineq_constraints_with_wildcard = ineq_constraints_with_wildcard
        self.parallelize = parallelize
        
    def fit(self):
        """
        Fits the model and generates reconciled forecasts for the input 
        dataframe subject to defined constraints.
        """

        df = self.df
        forecaster = self.forecaster
        constraints_with_wildcard = self.constraints_with_wildcard
        ineq_constraints_with_wildcard = self.ineq_constraints_with_wildcard
        parallelize = self.parallelize
        
        # modify inputs into machine-friendly shape
        df0, all_cells, unknown_cells, known_cells, islands = OrganizeCells(df)
        C,d = StringToMatrixConstraints(df0.T.stack(),
                                        all_cells,
                                        unknown_cells,
                                        known_cells,
                                        constraints_with_wildcard)
        C,d = AddIslandsToConstraints(C,d,islands)
        C_ineq,d_ineq = StringToMatrixConstraints(df0.T.stack(),
                                                  all_cells,
                                                  unknown_cells,
                                                  known_cells,
                                                  ineq_constraints_with_wildcard)
        # 1st stage forecast and its model
        df1,df1_model = FillAllEmptyCells(df0,forecaster,parallelize=parallelize)

        # get pseudo out-of-sample prediction, true values, and prediction models
        pred,true,model = GenPredTrueData(df0,forecaster,parallelize=parallelize)
        
        # break dataframe into list of time series
        ts_list,pred_list,true_list = BreakDataFrameIntoTimeSeriesList(df0,df1,pred,true)
        
        # get parts for reconciliation
        y1 = GenVecForecastWithIslands(ts_list,islands)
        W,shrinkage = GenWeightMatrix(pred_list, true_list)
        smoothness = GenLamstar(pred_list,true_list)
        Phi = GenSmoothingMatrix(W,smoothness)

        # 2nd stage forecast
        y2 = Reconciliation(y1,W,Phi,C,d,C_ineq,d_ineq)
        
        # reshape vector y2 into df2
        y2 = y2.T.stack(future_stack=True)
        y2.index = y2.index.droplevel(level=0)
        df2 = df0.copy()
        df2.update(y2,overwrite=False) # fill only nan cells of df0
        
        self.df0 = df0
        self.C = C
        self.d = d
        self.islands=islands
        
        self.df1 = df1
        self.df1_model = df1_model
        
        self.pred = pred
        self.true = true
        self.model = model
        self.ts_list = ts_list
        self.pred_list = pred_list
        self.true_list = true_list
        self.y1 = y1
        self.W = W
        self.Phi = Phi
        self.shrinkage = shrinkage
        self.smoothness = smoothness
        
        self.y2 = y2
        self.df2 = df2
        
        return self.df2

def OrganizeCells(df:pd.DataFrame):
    """Organize raw input data frame into known and unknown values, and identify 
    islands. Islands are values of known cells, preceded by unknown values.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with raw data.

    Returns
    -------
    df0 : pd.DataFrame
        Dataframe with island values replaced by nan.

    all_cells : pd.Series
        Series containing index of all cells in the input dataframe.

    unknown_cells : pd.Series
        Series containing index of cells whose values are to be forecasted.

    known_cells : pd.Series
        Series containing index of cells whose values are known.

    islands : pd.Series
        Series containing island values.
    """    
    def CleanIslands(df):
        """
        Separate islands from the raw input dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with raw data.

        Returns
        -------
        df_no_islands : pd.DataFrame
            Datafrane with island values replaced by nan.
        islands : pd.Series
            Series containing island values.
        """
        df_no_islands = df.copy() # to keep original df as it is
        col_with_islands = df.columns[df.isna().any()]
        coli_list = [df_no_islands.columns.get_loc(col) for col in col_with_islands]
        for coli in coli_list: # for col with na
            first_na_index = np.argwhere(df.iloc[:,coli].isna()).min()
            df_no_islands.iloc[first_na_index:,coli] = np.nan

        islands = df[df_no_islands.isna()].T.stack()
        return df_no_islands, islands
    
    # clean islands
    df0, islands = CleanIslands(df)

    # all cells in forecast horizon
    all_cells_index = df0.T.stack(future_stack=True).index
    all_cells = pd.Series([f'{a}_{b}' for a, b in all_cells_index],
                             index = all_cells_index)

    # unknown cells with nan 
    unknown_cells_index = df0.isna()[df0.isna()].T.stack().index
    unknown_cells = pd.Series([f'{a}_{b}' for a, b in unknown_cells_index],
                                 index = unknown_cells_index)
  
    # known cells
    known_cells_index = all_cells_index.difference(unknown_cells_index)
    known_cells = pd.Series([f'{a}_{b}' for a, b in known_cells_index],
                               index = known_cells_index)
    
    return df0, all_cells, unknown_cells, known_cells, islands


def StringToMatrixConstraints(df0_stacked:pd.DataFrame, # stack df0 to accomodate mixed frequency
                              all_cells:pd.Series,
                              unknown_cells:pd.Series,
                              known_cells:pd.Series,
                              constraints_with_wildcard:List[str] = [],
                              wildcard_string:str = '?'):
    """
    Convert equality constraints from list to matrix form gor horizons to 
    be forecasted (Cy = d, where C and d are dataframes containing the 
    linear constraints).

    Parameters
    ----------
    df0_stacked : pd.Series
        Stacked version of df0 (input  dataframe with islands removed), useful 
        for handling mixed frequency cases. 
    all_cells : pd.Series
        Series containing index of all cells in the input dataframe.
    unknown_cells : pd.Series
        Series containing index of cells whose values are to be forecasted.
    known_cells : pd.Series
        Series containing index of cells whose values are known..
    constraints_with_wildcard : str, optional
        String specifying equality constraints that have to hold. 
        The default is [].
    wildcard_string : str, optional
        String that is used as wildcard identifier in constraint. 
        The default is '?'.

    Returns
    -------
    C: pd.DataFrame
        Dataframe containing linear constraints in matrix form.
    d: pd.DataFrame
        Dataframe containing linear constraints in matrix form. (TBC)
    """
    def find_permissible_wildcard(constraints_with_wildcard):
        """Generate random letter to be used in constraints."""
        wild_card_length = 1
        candidate = ''.join(random.sample(ascii_lowercase,wild_card_length))
        while candidate in ''.join(constraints_with_wildcard):
            wild_card_length = wild_card_length + 1
            candidate = ''.join(random.sample(ascii_lowercase,wild_card_length))
        alphabet_wildcard = candidate
        return alphabet_wildcard


    def find_strings_to_replace_wildcard(constraint,var_list,wildcard):
        """"""
        varlist_regex = ['^' + str(v).replace(wildcard, '(.*)') + '$'
                         for v in sp.sympify(constraint).free_symbols]
        missing_string_set_list = []
        for w in varlist_regex:
            missing_string = []
            for v in var_list:
                match = re.compile(w).search(v)
                if match:
                    missing_string.append(match.group(1))
            missing_string_set_list.append(set(missing_string))
        missing_string_list = list(set.intersection(*missing_string_set_list))
        missing_string_list.sort()
        
        return missing_string_list


    def expand_wildcard(constraints_with_alphabet_wildcard,var_list,wildcard):
        """Expand constraints with wildcard to all forecast horizons.

        Parameters
        ----------
        constraints_with_alphabet_wildcard : str
            Linear equality constraints with wildcard string replaced 
            with alphabets.
        var_list : list
            List of indices of all cells (known and unknown) in raw dataframe.
        wildcard : str
            Alphabet which has replaced wildcard string in the constraints.

        Return
        ------
        expanded_constraints : list
            Expanded list of constraints over all time periods.
        """
        expanded_constraints = []
        for constraint in constraints_with_alphabet_wildcard:
            if wildcard not in constraint:
                expanded_constraints.append(constraint)
            else:
                missing_string_list = find_strings_to_replace_wildcard(constraint, var_list, wildcard)
                expanded_constraints += [constraint.replace(f'{wildcard}', m)
                                         for m in missing_string_list]
        return expanded_constraints
    
    # replace wildcard with alphabet to utilize sympy
    alphabet_wildcard = find_permissible_wildcard(constraints_with_wildcard)
    constraints_with_alphabet_wildcard = [c.replace(wildcard_string, alphabet_wildcard) \
                                          for c in constraints_with_wildcard]
    
    # expand constraints using all cells at forecast horizon
    constraints = expand_wildcard(constraints_with_alphabet_wildcard,
                                  var_list = all_cells.tolist(),
                                  wildcard = alphabet_wildcard)
    
    # obtain C_unknown by differentiating constraints wrt unknown cells with nan 
    A, b = sp.linear_eq_to_matrix(constraints, sp.sympify(unknown_cells.tolist()))
    C = pd.DataFrame(np.array(A).astype(float),
                     index = constraints,
                     columns = unknown_cells.index)
    nonzero_rows = (C != 0).any(axis=1)
    C = C.loc[nonzero_rows] # drop rows with all zeros
    
    # obtain d_unknown by substituting known cells
    known_cell_dict = pd.Series([df0_stacked.loc[idx] for idx in known_cells.index],
                                index = known_cells.tolist()).to_dict()
    d = pd.DataFrame(np.array(b.subs(known_cell_dict)).astype(float),
                             index = constraints)
    d = d.loc[nonzero_rows] # drop rows with all zeros in C
    
    return C,d


def AddIslandsToConstraints(C:pd.DataFrame,
                            d:pd.DataFrame,
                            islands):
    """
    Add island values into the matrix form equality constraints which have been
    constructed by ``StringToMatrixConstraints``.

    Parameters
    ----------
    C : pd.DataFrame
        Dataframe containing matrix of the linear constraints on the left side of
        equation Cy=d.
    d : pd.DataFrame
        Dataframe containing matrix of the linear constraints on the right side of
        equation Cy=d.
    islands : pd.Series
        Series containing island values to be introduced into linear equation.

    Returns
    -------
    C_aug : pd.DataFrame
        Dataframe containing the augmented C matrix, with island values incorporated.
    d_aug : pd.DataFrame
        Dataframe containing the augmented d vector, with island values incorporated.

    """
    C_aug_index = islands.index.union(C.index, sort=False) # singleton constraints prioritize over islands
    C_aug = pd.DataFrame(np.zeros([len(C_aug_index),len(C.columns)]),
                      index = C_aug_index,
                      columns = C.columns)
    d_aug = pd.DataFrame(np.zeros([len(C_aug_index),1]),
                      index = C_aug_index)
    for idx in islands.index:
        C_aug.loc[C_aug.index == idx,idx] = 1
        d_aug.loc[d_aug.index == idx] = islands.loc[idx]
    C_aug.update(C)
    d_aug.update(d)
    
    return C_aug,d_aug


def FillAnEmptyCell(df,row,col,forecaster):
    """Generate a forecast for a given cell based on the latest known value 
    for the given column (variable) and using the predefined forecasting pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing known values of all variables and nan for 
        unknown values.
    row : str
        Row index of cell to be forecasted.
    col : str
        Column index of cell to be forecasted. 
    forecaster : pipeline (?)
        Forecasting pipeline to be used for generating forecast.

    Returns
    -------
    y_pred : double
        Forecasted value of the variable for the given horizon.
    forecaster : 
         

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sktime.forecasting.compose import YfromX
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> from sklearn.linear_model import ElasticNetCV
    >>> n = 30
    >>> p = 2
    >>> df = pd.DataFrame(np.random.sample([n,p]),
    >>>                   columns=['a','b'],
    >>>                   index=pd.date_range(start='2000',periods=n,freq='YE').year)
    >>> df.iloc[-5:,:1] = np.nan
    >>> row = df.index[-1]
    >>> col = df.columns[0]
    >>> forecaster = YfromX(ElasticNetCV())
    >>> y_pred, forecaster = FillAnEmptyCell(df,row,col,forecaster)
    
    """
    warnings.filterwarnings('ignore', category=UserWarning)

    # last historical data and forecast horizon in num
    T = np.argwhere(df.loc[:,col].isna()).min() -1 
    h = np.where(df.index==row)[0][0] - T
    
    y = df.iloc[:T,:].loc[:,[col]]
    
    X = df.iloc[:T+h].drop(columns = [col]).dropna(axis=1) 
    X_train = X.iloc[:T,:]
    X_pred  = X.iloc[T:,:]
    
    y_pred  = forecaster.fit(y=y,X=X_train,fh=h).predict(X=X_pred)
    
    return y_pred, forecaster



def FillAllEmptyCells(df,forecaster,parallelize = True):
    """
    Generate forecasts for all unknown cells in the supplied dataframe.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe containing known values of all variables and nan for 
        unknown values.
    forecaster : 

    parrallelize : boolean
        Indicate whether paralellization should be employed for generating 
        first step forecasts 

    Return
    ------
    df1: pd.DataFrame
        Dataframe with all known cells, as well as unknown cells filled in by
        one-step forecasts.
    df1_model: pd.Dataframe
        Dataframe with all known cells, with unknown cells containing details 
        of the forecaster used for generating forecast of that cell.
    
    Examples
    -------- 
    >>> n = 30
    >>> p = 2
    >>> df = pd.DataFrame(random.sample([n,p]),
    >>>                   columns=['a','b'],
    >>>                   index=pd.date_range(start='2000',periods=n,freq='YE').year)
    >>> df.iloc[-5:,:1] = np.nan
    >>> def DefaultForecaster():
    >>>     return YfromX(ElasticNetCV(max_iter=5000))
    >>> df1,df1_models = FillAllEmptyCells(df,DefaultForecaster())
    
    """

    # get indices of all np.nan cells
    na_cells = [(df.index[rowi],df.columns[coli]) for rowi,coli in np.argwhere(df.isna())]
    
    # apply dask
    if parallelize == True:

        start = time()
        results = dask.compute(*[delayed(FillAnEmptyCell)(df,row,col,copy.deepcopy(forecaster)) 
                                  for (row,col) in na_cells],
                               scheduler = 'processes')
        end = time()
        print('Dask filled',len(results),'out-of-sample cells:',round(end-start,3),'seconds')
        
    else:
        start = time()
        results = [FillAnEmptyCell(df,row,col,forecaster) for row,col in na_cells]
        end = time()
        print('Forecast',len(results),'cells:',round(end-start,3),'seconds')
        
    # fill empty cells
    df1 = df.copy()
    df1_models = df.copy().astype(object)
    for idx, rowcol in enumerate(na_cells):
        df1.loc[rowcol] = results[idx][0].iloc[0,0]
        df1_models.loc[rowcol] = results[idx][1]
    
    return df1, df1_models


def GenPredTrueData(df,forecaster,n_sample=5,parallelize=True):
    """
    Generate in-sample forecasts from existing data by constructing 
    pseudo-historical datasets.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with all known as well as unknown values.
    forecaster : pipeline (?)
        Forecasting pipeline to be used
    n_sample : int, optional
        Number of horizons for which in-sample forecasts are generated.
        The default is 5.
    parallelize : boolean, optional
        Indicate whether parallelization should be used. The default is True.

    Returns
    -------
    pred : pd.DataFrame
        Dataframe with in-sample predictions generated using pseudo-historical 
        datasets. Dimensions are n_sample x n_sample.
    true : pd.DataFrame
        Dataframe with actual values of the variable corresponding to predicted
        values contained in pred.
    model : pd.DataFrame
        Dataframe with information on the models used for generating each
        forecast.

    """
  
    # last historical data and length of forecast horizon
    T = min(np.argwhere(df.isna())[:,0]) - 1
    h = max(np.argwhere(df.isna())[:,0]) - T
        
    # create pseudo historical dataframes and their na cells
    df_list = [df.shift(-h-n).mask(df.shift(-h-n).notna(),df).iloc[:-h-n,:] \
              for n in range(n_sample)]
    
    # unpack all the na cells for pseudo historical dataframes to use dask
    tasks = [(dfi,df.index[rowi],df.columns[coli]) \
             for dfi,df in enumerate(df_list) \
             for (rowi,coli) in np.argwhere(df.isna())]
    
    if parallelize == True:
        start = time()
        results = dask.compute(*[delayed(FillAnEmptyCell)(df_list[dfi],row,col,copy.deepcopy(forecaster)) \
                                      for (dfi,row,col) in tasks],
                               scheduler='processes') # processes, multiprocesses, threads won't work
        end = time()
        print('Dask filled',len(results),'in-sample cells:',round(end-start,3),'seconds')
    else:
        start = time()
        results = [FillAnEmptyCell(df_list[dfi],row,col,forecaster) \
                    for (dfi,row,col) in tasks]
        end = time()
        print('Fill',len(results),'in-sample cells:',round(end-start,3),'seconds')

    # repackage results by filling na of df_list
    filled_list = copy.deepcopy(df_list)
    model_list = [df.astype(object) for df in copy.deepcopy(df_list)]
    for task_idx,task in enumerate(tasks):
        dfi,row,col = task
        filled_list[dfi].loc[row,col] = results[task_idx][0].iloc[0,0]
        model_list[dfi].loc[row,col] = results[task_idx][1]
    
    # reduce n samples into a dataframe
    colname = df.isna()[df.isna()].T.stack().index
    idxname = pd.Index([df_list[n].index[np.argwhere(df_list[n].isna())[:,0].min()] 
                        for n in range(n_sample)],
                       name = 'LastData')
    pred = pd.DataFrame([filled_list[n][df_list[n].isna()].T.stack().values 
                         for n in range(n_sample)],index=idxname,columns=colname)
    model = pd.DataFrame([model_list[n][df_list[n].isna()].T.stack().values 
                         for n in range(n_sample)],index=idxname,columns=colname)    
    true = pd.DataFrame([df[df_list[n].isna()].T.stack().values 
                         for n in range(n_sample)],index=idxname,columns=colname)
    
    return pred,true,model

def BreakDataFrameIntoTimeSeriesList(df0,df1,pred,true):
    """ Transform relevant dataframes into lists for ensuing reconsiliation step.

    Parameters
    ----------
    df0 : pd.DataFrame
        Dataframe with all known and unknown values, without any islands.
    df1 : pd.DataFrame
        Dataframe with unknown values as well as islands filled in with 
        first step forecasts.
    pred : pd.DataFrame
        Dataframe with in-sample predictions generated using pseudo-historical 
        datasets, output from ``GenPredTrueData``.
    true : TYPE
        DESCRIPTION.

    Returns
    -------
    ts_list : list
        List containing all first step out of sample forecasts.
    pred_list : list
        List of dataframes, with each dataframe containing in-sample forecasts
        for one variable.
    true_list : list
        List of dataframes, with each dataframe containing the actuall values
        for a variable corresponding to in-sample predictions stored in
        pred_list.

    """
    ts_list = [df1[df0.isna()].loc[:,col:col].dropna().T.stack() for col in df0.columns[df0.isna().any()]]
    pred_list = [pred.loc[:,ts.index] for ts in ts_list]
    true_list = [true.loc[:,ts.index] for ts in ts_list]
    
    return ts_list,pred_list,true_list

def HP_matrix(size):
    """
    

    Parameters
    ----------
    size : TYPE
        DESCRIPTION.

    Returns
    -------
    F : TYPE
        DESCRIPTION.

    """
    if size >=2:
        D = np.zeros((size-2, size))
        for i in range(size-2):
            D[i, i] = 1
            D[i, i+1] = -2
            D[i, i+2] = 1
        F = D.T @ D
    elif size == 1:
        F = np.zeros([1,1])
    return F
    

def GenVecForecastWithIslands(ts_list,islands):
    """ Overwrite forecasted values for islands with known island value.
    
    Parameters
    ----------
    ts_list : list
        List of all first step forecasted values.
    islands : pd.Series
        Series containing island values.

    Returns
    -------
    y1 : pd.Series
        List of forecasted values with island values incorporated.

    """
    try:
        y1 = pd.concat(ts_list,axis=0)
    
    except: # only used in mixed-freq, pd.concat cann't process 4 mix-freq series
        y1 = ConcatMixFreqMultiIndexSeries(ts_list,axis=0)
        
    y1.update(islands)
        
    return y1


def GenWeightMatrix(pred_list,true_list,method='oas'):
    """ Generate weighting matrix based on in-sample forecasts and actual values.    

    Parameters
    ----------
    pred_list : list
        List of dataframes, with each dataframe containing in-sample forecasts
        for one variable..
    true_list : Tlist
        List of dataframes, with each dataframe containing the actual values
        for a variable corresponding to in-sample predictions stored in
        pred_list.
    method : str, optional
        Type of *Fill in*, with options of identity, oas and oasd. The default is 'oas'.

    Returns
    -------
    W : pd.DataFrame
        Weighting matrix to be used for reconciliation.
    shrinkage: float
        Shrinkage parameter associated with the weight. Nan in case identity
        is selected as method.

    """
    fe_list = [pred_list[i]-true_list[i] for i in range(len(pred_list))]
    
    try: # fe: sample size x vairables
        fe = pd.concat(fe_list,axis=1)
        
    except: # only used in mixed-freq, pd.concat cann't process 4 mix-freq series
        
        fe = ConcatMixFreqMultiIndexSeries(fe_list,axis=1)

    # sample covariance
    n_samp = fe.shape[0]
    n_vars = fe.shape[1]
    sample_cov = fe.cov()
    
    if method == 'identity':
        W = pd.DataFrame(np.eye(sample_cov.shape[0]),index=sample_cov.index,columns=sample_cov.columns)
        return W, np.nan
    
    if method == 'oas':    
        from sklearn.covariance import OAS
        oas = OAS().fit(fe.values)
        W = pd.DataFrame(oas.covariance_,index=sample_cov.index,columns=sample_cov.columns)
        rho = oas.shrinkage_
        return W, rho
    
    if method == 'oasd':
        if n_vars>=2:
            # shrinkage target
            diag = np.diag(np.diag(sample_cov))
            
            # shrinkage parameter
            numerator = np.trace(sample_cov @ sample_cov) - np.trace(diag @ diag)
            denominator = np.trace(sample_cov @ sample_cov) + np.trace(sample_cov) ** 2 - 2 * np.trace(diag @ diag)
            phi = numerator / denominator
            rho = min([1 / (n_samp * phi), 1])
            
            # shrink covariance matrix
            W = (1-rho) * sample_cov + rho * diag
        elif n_vars==1:
            W = sample_cov
            rho = np.nan
        return W, rho
    
    if method == 'monotone diagonal':
        if n_vars>=2:
            diag = pd.Series(np.diag(sample_cov),
                             index=sample_cov.index)
            W = pd.DataFrame(np.diag(diag.groupby(level=0).cummax()),
                             index = sample_cov.index,
                             columns=sample_cov.columns)
        elif n_vars==1:
            W = sample_cov
            rho = np.nan
        return W,np.nan


def GenLamstar(pred_list,true_list,empirically=True,default_lam=6.25):
    """
    Calculate the smoothness parameter (lambda) associated with each variable 
    being forecasted. 
    

    Parameters
    ----------
    pred_list : list
        List of dataframes, with each dataframe containing in-sample forecasts
        for one variable..
    true_list : Tlist
        List of dataframes, with each dataframe containing the actual values
        for a variable corresponding to in-sample predictions stored in
        pred_list.
    empirically : boolean, optional
        Indicate whether lambda should be calculated emperically, or use
        commonly used values from the literature. The default is True.
    default_lam : float, optional
        The value of lambda to use if none is provided. The default is 6.25.

    Returns
    -------
    lamstar : Series
        Series containing smoothing parameters to be used for each variable.

    """
    # index of time series to deal with mixed-frequency
    tsidx_list = [df.columns for df in pred_list]
    
    # box to store lamstar, columsn are the index of time series
    try: # extract freq info if available
        freq_list = [tsidx.get_level_values(1).freqstr[0] for tsidx in tsidx_list]
        ly = 100
        lambda_dict={'Y':ly,
                    'Q':ly*(4**2),
                    'M':ly*(12**2),
                    'W':ly*(52**2),
                    'D':ly*(365**2),
                    'H':ly*((365*24)**2),
                    'T':ly*((365*24*60)**2),
                    'S':ly*((365*24*60*60)**2)}
        lamstar = pd.Series( [lambda_dict[item].astype(float) for item in freq_list],
                            index = tsidx_list)
    except:
        lamstar = pd.Series( np.ones(len(tsidx_list)) * default_lam, 
                            index = tsidx_list)
    
    # optimal lambda
    if empirically == True:
        loss_fn = lambda x,T,yt,yp: \
            (yt - inv(np.eye(T) + x * HP_matrix(T)) @ yp).T @ \
            (yt - inv(np.eye(T) + x * HP_matrix(T)) @ yp)
        for tsidxi, tsidx in enumerate(tsidx_list):
            y_pred = pred_list[tsidxi]
            y_true = true_list[tsidxi]
            T = len(tsidx)
            obj = lambda x: np.mean([loss_fn(x,T,y_true.iloc[i:i+1,:].T.values,
                                             y_pred.iloc[i:i+1,:].T.values) \
                                 for i in range(y_pred.shape[0])])
            constraint = {'type': 'ineq', 'fun': lambda lam: lam}
            result = scipy.optimize.minimize(obj,0,constraints=[constraint])
            lamstar.iloc[tsidxi] = result.x[0]
    return lamstar


def GenSmoothingMatrix(W,lamstar):
    """
    Generate symmetric smoothing matrix using optimal lambda and weighting matrix.

    Parameters
    ----------
    W : pd.DataFrame
        Dataframe containing the weighting matrix.
    lamstar : pd.Series
        Series containing smoothing parameters to be used for each variable.

    Returns
    -------
    Phi : pd.DataFrame
        Dataframe containing the smoothing matrix.

    """
    lam = lamstar/[np.diag(W.loc[tsidx,tsidx]).min() 
                   for tsidx in lamstar.index]
    Phi_np = block_diag(*[lam.iloc[tsidxi] * HP_matrix(len(tsidx)) 
                       for tsidxi,tsidx in enumerate(lam.index)])
    Phi = pd.DataFrame(Phi_np,index=W.index,columns=W.columns)
    return Phi
    

def Reconciliation(y1,W,Phi,C,d,C_ineq=None,d_ineq=None):
    """
    Reconcile first step forecasts to satisfy constraints, with smoothening
    parameters implemented.

    Parameters
    ----------
    y1 : Series
        Series of all forecasted and island values.
    W : pd.DataFrame
        Dataframe containing the weighting matrix.
    Phi : pd.DataFrame
        Dataframe containing the smoothing matrix.
    C : pd.DataFrame
        Dataframe containing matrix of the linear constraints on the left side of
        the equality constraint Cy=d.
    d : pd.DataFrame
        Dataframe containing matrix of the linear constraints on the right side of
        the equality constraint Cy=d.
    C_ineq : pd.DataFrame, optional
        Dataframe containing matrix of the linear constraints on the left side of
        the equality constraint Cy <= d. The default is None.
    d_ineq : TYPE, optional
        Dataframe containing matrix of the linear constraints on the left side of
        the equality constraint Cy <= d. The default is None.

    Returns
    -------
    y2 : pd.DataFrame
        Dataframe containing the final reconciled forecasts for all variables.

    """
    assert((y1.index == W.index).all())
    assert((y1.index == Phi.index).all())
    assert((y1.index == C.columns).all())
    assert((C.index  == d.index).all())
    
    def DropLinDepRows(C_aug,d_aug):
        
        C = C_aug.values
        
        # Convert the matrix to a SymPy Matrix
        sympy_matrix = sp.Matrix(C)
        
        # Compute the RREF and get the indices of linearly independent rows
        rref_matrix, independent_rows = sympy_matrix.T.rref()
            
        # Extract the independent rows
        independent_rows = list(independent_rows)
                
        # dependent rows
        all_rows = set(range(C.shape[0]))
        dependent_rows = list(all_rows - set(independent_rows))
       
        C = C_aug.iloc[independent_rows, :]
        d = d_aug.iloc[independent_rows,:]
        
        if dependent_rows != []:
            print('Constraints are linearly dependent. The following constraints are dropped.',
                  C_aug.index[dependent_rows])
        return C,d
    
    # keep lin indep rows
    C,d = DropLinDepRows(C,d)
    
    # reconcile with np.array
    W_inv = inv(W)
    denom = inv(W_inv + Phi)
    Cn = C.values
    dn = d.values
    CdC_inv = inv(Cn @ denom @ Cn.T) # removing linearly dependent rows to use inv doesn't change results much
    
    In = np.eye(len(y1))
    y1n = y1.values.reshape(-1,1)
    y2n = ( In - denom @ Cn.T @ CdC_inv @ Cn ) @ denom @ W_inv @ y1n + \
        denom @ Cn.T @ CdC_inv @ dn
    
    if C_ineq is not None and C_ineq.shape[0]>0:
        
        C_ineq,d_ineq = DropLinDepRows(C_ineq,d_ineq)
        
        # augment C_ineq, d_ineq to be compatible with y1
        C_ineq_aug = pd.DataFrame(np.zeros([len(C_ineq.index),len(y1)]),
                                  index = C_ineq.index,
                                  columns = y1.index)
        C_ineq_aug.update(C_ineq)
        d_ineq_aug = pd.DataFrame(np.zeros([len(d_ineq.index),1]),
                                  index = d_ineq.index)
        d_ineq_aug.update(d_ineq)
        Cn_ineq = C_ineq_aug.values
        dn_ineq = d_ineq_aug.values
    
        # use CVXPY to solve numerically
        P = W_inv + Phi
        q = - 2 * W_inv @ y1n
        x = cp.Variable([len(y1),1])
        objective = cp.Minimize(cp.quad_form(x,P,assume_PSD=True) + q.T @ x)
        constraints = [Cn @ x == dn, Cn_ineq @ x <= dn_ineq]
        prob = cp.Problem(objective,constraints)
        prob.solve()
        y2n = x.value
        
        if y2n is None:
            import warnings
            warnings.warn("Reconciliation failed. Feasible sets might be empty.")
            
    # put reconciled y2 back to dataframe
    y2 = pd.DataFrame(y2n,index=y1.index)

    return y2


def example1(): # no constraints

    # load data
    from sktime.datasets import load_macroeconomic
    df_true = load_macroeconomic().iloc[:,:5]
    
    # input dataframe
    df = df_true.copy()
    fh = 5
    df.iloc[-fh:,0] = np.nan
    
    # apply MFF
    m = MFF(df,constraints_with_wildcard=[])
    df2 = m.fit()
    df0 = m.df0
    df1 = m.df1
    df1_model = m.df1_model
    smoothness = m.smoothness
    shrinkage = m.shrinkage
    
    # plot results
    t0 = -30
    ax = df0.iloc[t0:,0].plot(label='df0')
    df1.iloc[t0:,0].plot(ax=ax,label='df1')
    df2.iloc[t0:,0].plot(ax=ax,label='df2')
    df_true.iloc[t0:,0].plot(ax=ax,label='df_true')
    ax.axvline(x = df0.index[-fh])
    ax.legend()
    
    print('smoothness',smoothness.values)
    print('shrinkage',np.round(shrinkage,3))
    for ri,ci in np.argwhere(df.isna()):
        print(df1_model.index[ri],
              df1_model.columns[ci],
              df1_model.iloc[ri,ci].best_params_)

        
# example 2: with constraints
def example2():
    
    # create data
    n = 30
    p = 3
    fh = 5
    df_true = pd.DataFrame(np.random.rand(n,p),
                      columns=[f'{L}{i}' for i in range(int(np.ceil(p/26))) for L in ascii_uppercase][:p],
                      index=pd.date_range(start='2000',periods=n,freq='YE').year
                      )
    df_true.iloc[:,-1] = df_true.iloc[:,:-1].sum(axis=1)
    df = df_true.copy()
    df.iloc[-fh:,:np.ceil(p/2).astype(int)] = np.nan
    df.iloc[-1,0] = df_true.iloc[-1,0] # island
    #df.iloc[-fh,-1] = df.iloc[:,-1].mean()
    # df.iloc[-3,1] = df_true.iloc[-3,1] # island
    constraints_with_wildcard = ['A0?+B0?-C0?']
    #ineq_constraints_with_wildcard = ['A0?-0.5'] # A0 <=0.5 for all years
    
    # fit data
    m = MFF(df,constraints_with_wildcard = constraints_with_wildcard)
    df2 = m.fit()
    df0 = m.df0
    df1 = m.df1
    df1_model = m.df1_model
    shrinkage = m.shrinkage
    smoothness = m.smoothness
    W = m.W
    for ri,ci in np.argwhere(df.isna()):
        print(df1_model.index[ri],
              df1_model.columns[ci],
              df1_model.iloc[ri,ci].best_params_)
    
    
    import matplotlib.pyplot as plt
    plt.figure()
    t0 = -20
    plt.subplot(2,1,1)
    ax = df0.iloc[t0:,0].plot(label='df0')
    df1.iloc[t0:,0].plot(ax=ax,label='df1')
    df2.iloc[t0:,0].plot(ax=ax,label='df2')
    df_true.iloc[t0:,0].plot(ax=ax,label='df_true')
    ax.axvline(x = df0.index[-fh])
    
    plt.subplot(2,1,2)
    ax = df0.iloc[t0:,1].plot(label='df0')
    df1.iloc[t0:,1].plot(ax=ax,label='df1')
    df2.iloc[t0:,1].plot(ax=ax,label='df2')
    df_true.iloc[t0:,1].plot(ax=ax,label='df_true')
    ax.axvline(x = df0.index[-fh],label='fh=1')
    ax.legend(loc='lower left')
    
    print('smoothness',smoothness.values)
    print('shrinkage',np.round(shrinkage,3))
    
    # confirm constraints
    assert(np.isclose(df2['A0']+df2['B0']-df2['C0'],0).all())

    

    
#%% MFF mixed freq
class MFF_mixed_freqency:
    
    def __init__(self,
                 df_dict,
                 forecaster = DefaultForecaster(),
                 constraints_with_wildcard=[],
                 ineq_constraints_with_wildcard=[]):
        
        self.df_dict = df_dict
        self.forecaster=forecaster
        self.constraints_with_wildcard=constraints_with_wildcard
        self.ineq_constraints_with_wildcard=ineq_constraints_with_wildcard
        
    def fit(self):
        df_dict = self.df_dict
        forecaster=self.forecaster
        constraints_with_wildcard=self.constraints_with_wildcard
        ineq_constraints_with_wildcard=self.ineq_constraints_with_wildcard
        
        # create constraints
        freq_order = ['Y', 'Q', 'M', 'W', 'D', 'H', 'T', 'S']    
        lowest_freq = freq_order[min([freq_order.index(k) for k in df_dict.keys()])]

        df0_list = []
        all_cells_list = []
        unknown_cells_list = []
        known_cells_list = []
        islands_list = []
        for k in df_dict.keys():
            df0_k, all_cells_k, unknown_cells_k, known_cells_k, islands_k = \
                OrganizeCells(df_dict[k])
            df0_list.append(df0_k)
            all_cells_list.append(all_cells_k)
            unknown_cells_list.append(unknown_cells_k)
            known_cells_list.append(known_cells_k)
            islands_list.append(islands_k)

        df0_stacked = ConcatMixFreqMultiIndexSeries([df0.T.stack() for df0 in df0_list], axis=0)
        all_cells = pd.concat(all_cells_list,axis=0)
        unknown_cells = pd.concat(unknown_cells_list,axis=0)
        known_cells = pd.concat(known_cells_list,axis=0)
        islands = pd.concat(islands_list,axis=0)

        C,d = StringToMatrixConstraints(df0_stacked,
                                        all_cells,
                                        unknown_cells,
                                        known_cells,
                                        constraints_with_wildcard)

        # combine all frequncies into the lowest frequency dataframe
        df0wide_list = []
        df0wide_colflat_list = []
        for df in df0_list:
            
            df0 = df.copy() # don't want to change df0_list
            df0_freq = df0.index.freqstr[0] 
            
            if df0_freq == lowest_freq:
                df0wide_freq = df0.copy()
                df0wide_colfat_freq = pd.Series(df0wide_freq.columns,
                                              index = df0wide_freq.columns)
                
            else:
                index_freq = df0.index.asfreq(lowest_freq)
                col_freq = df0_freq + get_freq_of_freq(df0.index,df0_freq).astype(str)
                df0.index = pd.MultiIndex.from_arrays([index_freq,col_freq])
                df0wide_freq = df0.unstack()
                df0wide_colfat_freq = pd.Series(df0wide_freq.columns.map('_'.join), 
                                                   index = df0wide_freq.columns)
                
            df0wide_list.append(df0wide_freq)
            df0wide_colflat_list.append(df0wide_colfat_freq)

        df0wide = pd.concat(df0wide_list,axis=1)
        df0wide_col = df0wide.columns
        df0wide_colflat = pd.concat(df0wide_colflat_list)

        # 1st step forecast
        df0wide.columns = df0wide_colflat.values.tolist() # colname has to be single index
        df1wide,df1wide_model = FillAllEmptyCells(df0wide,forecaster)
        predwide,truewide,modelwide = GenPredTrueData(df0wide,forecaster)

        # get df1_list by breaking wide dataframe into different frequencies
        df1_list = []
        for df0i,df0 in enumerate(df0_list):
            if df0.index.freqstr[0] == lowest_freq:
                df1_freq = df0.copy()
                df1_freq.update(df1wide.loc[:,df0wide_colflat_list[df0i].values])    
            else:
                df1wide_freq = df1wide.loc[:,df0wide_colflat_list[df0i].values]
                df1wide_freq.columns = pd.MultiIndex.from_tuples(df0wide_colflat_list[df0i].index)
                df1_freq = df0wide_list[df0i].copy().stack(future_stack=True) # storage
                df1_freq.update(df1wide_freq.stack(future_stack=True))
                df1_freq.index = df0_list[df0i].index
            
            df1_list.append(df1_freq)

        # get pred_list, true_list by breaking dataframes into different frequencies
        pred_allfreq = []
        true_allfreq = []
        for df0i,df0 in enumerate(df0_list):
            
            # get nan cells
            df0wide_freq = df0wide_list[df0i].copy()
            df0wide_freq.columns = df0wide_colflat_list[df0i].values
            na_cells = df0wide_freq.isna()[df0wide_freq.isna()].T.stack().index
            
            # slice predwide
            pred_freq = predwide.loc[:,na_cells]
            true_freq = truewide.loc[:,na_cells]
            
            if df0.index.freqstr[0] != lowest_freq:
                    
                # reshape colname multiindex of (var_freq,lowestfreq) to var_lowestfreqfreq
                colflat = pred_freq.columns
                var_list = [v[:v.rfind('_')] for v in colflat.get_level_values(0)]
                freq_list = [v[v.rfind('_')+1:] for v in colflat.get_level_values(0)]
                lowest_freq_list = colflat.get_level_values(-1).astype(str)
                original_time = pd.PeriodIndex([lowest_freq_list[i]+freq_list[i] for i in range(len(colflat))],
                                               freq=df0.index.freq)
                pred_freq_colname = pd.MultiIndex.from_arrays([var_list,original_time])
                pred_freq.columns = pred_freq_colname
                true_freq.columns = pred_freq_colname
            
            # change col order
            pred_freq = pred_freq.loc[:,df0.isna()[df0.isna()].T.stack().index]
            true_freq = true_freq.loc[:,pred_freq.columns]
            
            # append pred, true for each frequency
            pred_allfreq.append(pred_freq)
            true_allfreq.append(true_freq)

        # break dataframes in to lists
        ts_list = []
        pred_list = []
        true_list = []
        for df0i,df0 in enumerate(df0_list):
            ts_list_freq,pred_list_freq,true_list_freq = BreakDataFrameIntoTimeSeriesList(
                df0,df1_list[df0i],pred_allfreq[df0i],true_allfreq[df0i])
                
            ts_list += ts_list_freq
            pred_list += pred_list_freq
            true_list += true_list_freq
              
        # get parts for reconciliation
        #islands_list_all_freq = pd.concat(islands_list)

        y1 = GenVecForecastWithIslands(ts_list,islands)
        W,shrinkage = GenWeightMatrix(pred_list, true_list)
        smoothness = GenLamstar(pred_list,true_list)
        Phi = GenSmoothingMatrix(W,smoothness)
        
        y2 = Reconciliation(y1,W,Phi,C,d)

        # reshape vector y2 into df2
        y2 = y2.T.stack(future_stack=True)
        y2.index = y2.index.droplevel(level=0)
        df2_list=[]
        for df0 in df0_list:
            df2_freq = df0.copy()
            df2_freq.update(y2,overwrite=False) # fill only nan cells of df0
            df2_list.append(df2_freq)


        self.df0_list = df0_list
        self.df1_list = df1_list
        self.df2_list = df2_list
        return self.df2_list
    
# used only in mixed freq case, pd.concat doesn't work for more than 4 mix-freq series
# doesn't work when there are more than 3 freq!
def ConcatMixFreqMultiIndexSeries(df_list,axis):
    
    try:
        return pd.concat(df_list,axis=axis)
    except:
        
        if axis == 0:
            
            # concat by freq
            freqs = [df.index.get_level_values(1).freqstr[0] for df in df_list]
            seen = set()
            freq_unique = [x for x in freqs if not (x in seen or seen.add(x))]
            dflong_list = []
            for k in freq_unique:
                df_list_k = [df for df in df_list if df.index.get_level_values(1).freqstr[0] == k]
                dflong_k = pd.concat(df_list_k,axis=0)
                dflong_list.append(dflong_k)
            
            dflong = pd.concat(dflong_list,axis=0)           
            return dflong
        
        if axis == 1:
            
            # concat by freq
            freqs = [df.columns.get_level_values(1).freqstr[0] for df in df_list]
            seen = set()
            freq_unique = [x for x in freqs if not (x in seen or seen.add(x))]
            dfwide_list = []
            for k in freq_unique:
                df_list_k = [df for df in df_list \
                             if df.columns.get_level_values(1).freqstr[0] == k]
                dfwide_k = pd.concat(df_list_k,axis=1)
                dfwide_list.append(dfwide_k)
            
            dfwide = pd.concat(dfwide_list,axis=1)
            return dfwide

def get_freq_of_freq(periodindex,freqstr):
    if freqstr == 'Y':
        return periodindex.year
    if freqstr == 'Q':
        return periodindex.quarter
    if freqstr == 'M':
        return periodindex.month
    if freqstr == 'W':
        return periodindex.week
    if freqstr == 'D':
        return periodindex.day
    if freqstr == 'H':
        return periodindex.hour
    if freqstr == 'T':
        return periodindex.min
    if freqstr == 'S':
        return periodindex.second


# example, mixed-frequency intra-inter-temporal constraints
def example3():
    
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    n = 120
    p = 3
    fhA = 5
    fhQ = 7
    dfQ_true = pd.DataFrame(np.random.rand(n,p),
                      columns=[f'{L}{i}' for i in range(int(np.ceil(p/26))) for L in ascii_uppercase][:p],
                      index=pd.period_range(start='2000-1-1',periods=n,freq='Q'))
    dfQ_true.iloc[:,-1] = dfQ_true.iloc[:,:-1].sum(axis=1)
    dfA_true = dfQ_true.groupby(dfQ_true.index.year).sum()
    dfA_true.index = pd.PeriodIndex(dfA_true.index,freq='Y')
    
    dfA = dfA_true.copy()
    dfA.iloc[-fhA:,:np.ceil(p/2).astype(int)] = np.nan
    
    dfQ = dfQ_true.iloc[:-12,:].copy()
    dfQ.iloc[-fhQ:,:np.ceil(p/2).astype(int)] = np.nan
    
    # inputs
    df_dict = {'Y':dfA,'Q':dfQ}
    constraints_with_wildcard = ['A0?+B0?-C0?','?Q1+?Q2+?Q3+?Q4-?']
    
    mff = MFF_mixed_freqency(df_dict,
                             constraints_with_wildcard=constraints_with_wildcard)
    df2_list = mff.fit()
    df1_list = mff.df1_list
    df0_list = mff.df0_list
    
    # plot results
    import matplotlib.pyplot as plt
    t0 = -30
    plt.subplot(2,1,1)
    ax = df0_list[1].iloc[t0:,0].plot(label='df0')
    df1_list[1].iloc[t0:,0].plot(ax=ax,label='df1')
    df2_list[1].iloc[t0:,0].plot(ax=ax,label='df2')
    dfQ_true.iloc[t0:,0].plot(ax=ax,label='df_true')
    ax.axvline(x = df0_list[1].index[-fhQ],label='fh=1')
    ax.legend(loc='lower left')
    
    plt.subplot(2,1,2)
    ax = df0_list[0].iloc[t0:,0].plot(label='df0')
    df1_list[0].iloc[t0:,0].plot(ax=ax,label='df1')
    df2_list[0].iloc[t0:,0].plot(ax=ax,label='df2')
    dfA_true.iloc[t0:,0].plot(ax=ax,label='df_true')
    ax.axvline(x = df0_list[0].index[-fhQ],label='fh=1')
    ax.legend(loc='lower left')
    
    # check constraints
    df2A = df2_list[0]
    df2Q = df2_list[1]
    df2A.eval('A0+B0-C0')
    (df2Q.resample('Y').sum()-df2A).dropna()
