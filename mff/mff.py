from scipy.linalg import block_diag
from dask import delayed
from numpy.linalg import inv
from sklearn.linear_model import ElasticNetCV
from sktime.forecasting.compose import YfromX
from sktime.forecasting.base import ForecastingHorizon
from time import time
import copy
import dask
import random
import re
import scipy
import cvxpy as cp
import numpy as np
import pandas as pd
import sympy as sp
from string import ascii_lowercase

class MFF:
    
    def __init__(self):
        pass
        
    def fit(self,df,constraints_with_wildcard=[],ineq_constraints_with_wildcard=[]):
        """
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        constraints_with_wildcard : TYPE, optional
            DESCRIPTION. The default is [].
        ineq_constraints_with_wildcard : TYPE, optional
            DESCRIPTION. The default is [].

        Returns
        -------
        TYPE
            DESCRIPTION.
            
        Examples
        -------
        T = 30
        n = 2
        k = 1
        fh = 5
        df_true = pd.DataFrame(np.random.rand(T,n+k),
                          columns=['A','B','c'],
                          #index=pd.date_range(start='2000',periods=T,freq='YE').year
                          )
        df_true.iloc[:,-1] = df_true.iloc[:,:-1].sum(axis=1)
        df = df_true.copy()
        df.iloc[-fh:,:n] = np.nan
        df.iloc[-1,0] = df_true.iloc[-1,0] # island
        constraints_with_wildcard = ['A0?+B0?-C0?']
        ineq_constraints_with_wildcard = ['A0?-0.5'] # A0 <=0.5 for all years

        mff = MFF()
        df2 = mff.fit(df,constraints_with_wildcard)

        mff = MFF()
        df2 = mff.fit(df,constraints_with_wildcard,
                      ineq_constraints_with_wildcard)
        
        """
        # put inputs into machine-friendly shape
        df0, all_cells, unknown_cells, known_cells, islands = OrganizeCellsInForecastHorizon(df)
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
        # 1st stage
        df1,df1_model = FillAllEmptyCells(df0)

        # get pseudo out-of-sample forecast and true values
        pred,true,model = GenPredTrueData(df0)
        
        # break dataframe into list of time series
        ts_list,pred_list,true_list = BreakDataFrameIntoTimeSeriesList(df0,df1,pred,true)
        
        # get parts for reconciliation
        y1,W,Phi,shrinkage,smoothness = GenReconciliationParts(ts_list,pred_list,true_list,islands)
        
        # 2nd stage
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

def OrganizeCellsInForecastHorizon(df):
    """
    

    Parameters
    ----------
    df : pd.DataFrame()
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    def CleanIslands(df):
        """
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.

        Returns
        -------
        df_no_islands : TYPE
            DESCRIPTION.
        islands : TYPE
            DESCRIPTION.

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


def StringToMatrixConstraints(df0_stacked, # stack df0 to accomodate mixed frequency
                              all_cells,
                              unknown_cells,
                              known_cells,
                              constraints_with_wildcard=[],
                              wildcard_string = '?'):
  
    def find_permissible_wildcard(constraints_with_wildcard):
        wild_card_length = 1
        candidate = ''.join(random.sample(ascii_lowercase,wild_card_length))
        while candidate in ''.join(constraints_with_wildcard):
            wild_card_length = wild_card_length + 1
            candidate = ''.join(random.sample(ascii_lowercase,wild_card_length))
        alphabet_wildcard = candidate
        return alphabet_wildcard


    def find_strings_to_replace_wildcard(constraint,var_list,wildcard):
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
    C = C.loc[(C != 0).any(axis=1)] # drop rows with all zeros
    
    # obtain d_unknown by substituting known cells
    known_cell_dict = pd.Series([df0_stacked.loc[idx] for idx in known_cells.index],
                                index = known_cells.tolist()).to_dict()
    d = pd.DataFrame(np.array(b.subs(known_cell_dict)).astype(float),
                             index = constraints)

    return C,d


def AddIslandsToConstraints(C,d,islands):

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


def init_forecaster():
    
    return YfromX(ElasticNetCV(max_iter=5000))


def FillAnEmptyCell(df,row,col,forecaster):
    """  
    import numpy as np
    import pandas as pd
    
    n = 30
    p = 2
    df = pd.DataFrame(random.sample([n,p]),
                      columns=['a','b'],
                      index=pd.date_range(start='2000',periods=n,freq='YE').year)
    df.iloc[-5:,:1] = np.nan
    
    row = df.index[-1]
    col = df.columns[0]
    forecaseter = YfromX(ElasticNetCV())
    y_pred, forecaster, y_hist, X_hist, X_pred = FillAnEmptyCell(df,row,col,forecaseter)
    
    """

    # last historical data and forecast horizon in num
    T = np.argwhere(df.loc[:,col].isna()).min() -1 
    h = np.where(df.index==row)[0][0] - T
    
    # extended data sets with lags
    df_lags = pd.concat([df.shift(lag) for lag in range(h+1)],axis=1)
    df_lags_colnames = df.columns
    for lag in range(1,h+1):
        df_lags_colnames = df_lags_colnames.append('L'+str(lag)+'_'+df.columns)
    df_lags.columns = df_lags_colnames
    
    # create train and pred sets
    y_hist = df_lags.iloc[h:T+1,:].loc[:,col]
    X_all_lags = df_lags.drop(columns=[col])
    X_hist = X_all_lags.iloc[h:T+1,:].loc[:,~X_all_lags.iloc[T+h,:].isna()]
    X_pred = X_all_lags.iloc[T+h:T+h+1,:].loc[:,~X_all_lags.iloc[T+h,:].isna()]
    
    # fit and predict forecaster
    fh = ForecastingHorizon(h,is_relative=True)
    y_pred = forecaster.fit_predict(y=y_hist,X=X_hist,fh=fh,X_pred=X_pred)
    
    # no lags
    # y_hist = df.loc[:,col].iloc[:T+1]
    # X_hist = df.loc[:,~df.isna().any()].iloc[:T+1,:]
    # X_pred = df.loc[:,~df.isna().any()].iloc[T+h:T+h+1,:]
    # y_pred = forecaster.fit(y=y_hist,X=X_hist,fh=fh).predict(X=X_pred)
    
    return y_pred, forecaster, y_hist, X_hist, X_pred

def FillAllEmptyCells(df,init_forecaster=init_forecaster, parallelize=False):
    """  
    n = 30
    p = 2
    df = pd.DataFrame(random.sample([n,p]),
                      columns=['a','b'],
                      index=pd.date_range(start='2000',periods=n,freq='YE').year)
    df.iloc[-5:,:1] = np.nan
    def init_forecaster():
        return YfromX(ElasticNetCV(max_iter=5000))
    df1,df1_models = FillAllEmptyCells(df,init_forecaster)
    
    """

    # get indices of all np.nan cells
    na_cells = [(df.index[rowi],df.columns[coli]) for rowi,coli in np.argwhere(df.isna())]
    
    # apply dask
    if parallelize == True:
        start = time()
        results = dask.compute(*[delayed(FillAnEmptyCell)(df,row,col,init_forecaster()) 
                                 for (row,col) in na_cells])
        end = time()
        print('Dask filled',len(results),'out-of-sample cells:',round(end-start,3),'seconds')
        
    else:
        start = time()
        results = [FillAnEmptyCell(df,row,col,init_forecaster()) for row,col in na_cells]
        end = time()
        print('Forecast',len(results),'cells:',round(end-start,3),'seconds')
        
    # fill empty cells
    df1 = df.copy()
    df1_models = df.copy().astype(object)
    for idx, rowcol in enumerate(na_cells):
        df1.loc[rowcol] = results[idx][0].values
        df1_models.loc[rowcol] = results[idx][1]
    
    return df1, df1_models


def GenPredTrueData(df,init_forecaster=init_forecaster,n_sample=5,parallelize=False):
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
    
    # apply dask
    if parallelize == True:
        start = time()
        results = dask.compute(*[delayed(FillAnEmptyCell)(df_list[dfi],row,col,init_forecaster()) \
                                     for (dfi,row,col) in tasks]) # processes, multiprocesses, threads won't work
        end = time()
        print('Dask filled',len(results),'in-sample cells:',round(end-start,3),'seconds')
    else:
        start = time()
        results = [FillAnEmptyCell(df_list[dfi],row,col,init_forecaster()) \
                   for (dfi,row,col) in tasks]
        end = time()
        print('Fill',len(results),'in-sample cells:',round(end-start,3),'seconds')

    # repackage results by filling na of df_list
    filled_list = copy.deepcopy(df_list)
    model_list = [df.astype(object) for df in copy.deepcopy(df_list)]
    for task_idx,task in enumerate(tasks):
        dfi,row,col = task
        filled_list[dfi].loc[row,col] = results[task_idx][0].values
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
    ts_list = [df1[df0.isna()].loc[:,col:col].dropna().T.stack() for col in df0.columns[df0.isna().any()]]
    pred_list = [pred.loc[:,ts.index] for ts in ts_list]
    true_list = [true.loc[:,ts.index] for ts in ts_list]
    
    return ts_list,pred_list,true_list

def HP_matrix(size):
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
    
def GenReconciliationParts(ts_list,pred_list,true_list,islands):
       
    def VectorizedForecastWithIslands(ts_list,islands):
        
        try:
            y1 = pd.concat(ts_list,axis=0)
        
        except: # only used in mixed-freq, pd.concat cann't process 4 mix-freq series
            
           y1 = ConcatMixFreqMultiIndexSeries(ts_list,axis=0)
            
        y1.update(islands)
            
        return y1
    
    def GenWeightMatrix(pred_list,true_list,method='oas'):
        
        fe_list = [pred_list[i]-true_list[i] for i in range(len(pred_list))]
        fe = pd.concat(fe_list,axis=1)

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
    
    def CalibrateLamstar(pred_list,true_list,empirically=False,default_lam=6.25):
        
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
            lamstar = pd.Series( [lambda_dict[item] for item in freq_list],
                                index = tsidx_list)
        except:
            lamstar = pd.Series( np.ones(len(tsidx_list)) * default_lam, 
                                index = tsidx_list)
        
        # optimal lambda
        if empirically == True:
            loss_fn = lambda x,T,y_true,y_pred: \
                (y_true - inv(np.eye(T) + x * HP_matrix(T)) @ y_pred)**2
            for tsidxi, tsidx in enumerate(tsidx_list):
                y_pred = pred_list[tsidxi]
                y_true = true_list[tsidxi]
                T = len(tsidx)
                obj = lambda x: sum([loss_fn(x,T,y_true.iloc[i:i+1,:].T.values,
                                                 y_pred.iloc[i:i+1,:].T.values) \
                                     for i in range(y_pred.shape[0])])
                constraint = {'type': 'ineq', 'fun': lambda lam: lam}
                result = scipy.optimize.minimize(obj,0,constraints=[constraint])
                lamstar.iloc[tsidxi] = result.x
            
        return lamstar
    
    
    def GenSmoothingMatrix(W,lamstar):
        lam = lamstar/[np.diag(W.loc[tsidx,tsidx]).min() 
                       for tsidx in lamstar.index]
        Phi_np = block_diag(*[lam.iloc[tsidxi] * HP_matrix(len(tsidx)) 
                           for tsidxi,tsidx in enumerate(lam.index)])
        Phi = pd.DataFrame(Phi_np,index=W.index,columns=W.columns)
        return Phi
    

    y1 = VectorizedForecastWithIslands(ts_list,islands)
    W,rho = GenWeightMatrix(pred_list, true_list)
    lamstar = CalibrateLamstar(pred_list,true_list)
    Phi = GenSmoothingMatrix(W,lamstar)
    
    return y1,W,Phi,rho,lamstar
    

def Reconciliation(y1,W,Phi,C,d,C_ineq=None,d_ineq=None):

    assert((y1.index==W.index).all())
    assert((y1.index==Phi.index).all())
    assert((y1.index==C.columns).all())
    
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

