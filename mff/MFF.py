# Disclaimer: Reuse of this tool and IMF information does not imply
# any endorsement  of the research and/or product. Any research presented
# should not be reported as representing the views of the IMF,
# its Executive Board, member governments.

from typing import List

import pandas as pd

from mff.utils import (
    DefaultForecaster,
    OrganizeCells,
    StringToMatrixConstraints,
    AddIslandsToConstraints,
    FillAllEmptyCells,
    GenPredTrueData,
    BreakDataFrameIntoTimeSeriesList,
    GenVecForecastWithIslands,
    GenWeightMatrix,
    GenLamstar,
    GenSmoothingMatrix,
    Reconciliation
    )

#%% MFF

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
    
    forecaster : BaseForecaster
        sktime BaseForecaster descendant
    
    constraints_with_wildcard : str, optional(default: None)
        Constraints that hold with equality. Constraints may include wildcard, 
        in which case constraints will be applied across all horizons, or
        may be defined for specified time periods.
     
    ineq_constraints_with_wildcard : str, optional(default: None)
        Inequality constraints, comparable to ``constraints_with_wildcard``.
        Constraints may include wildcard, in which case constraints will be
        applied across all horizons, or may be defined for specified time 
        periods.
       
    parallelize : boolean
        Indicate whether parallelization should be employed for generating the
        first step forecasts. Default value is `True`. 

    n_sample_split : int
        Number of windows to split data into training and testing sets for 
        generating matrix of forecast errors. Default is 5.

    shrinkage_method : str, optional(default: 'oas')
        Method to be used for shrinking sample covariance matrix. Default is 
        Oracle Shrinking Approximating Estimator ('oas'). Other options are
        oas, identity and monotone_diagonal.
    
    lamstar_empirically : boolean, optional(default: True)
        Indicate whether the smoothness paramatere lambda is to be calculated
        empirically 

    default_lam : float, optional(default: 6.25)
        Default value of lambda to be used; used when lamstar is not being estimated
        empirically.

    max_lam : float, optional(default: 129600)
        Maximum value of lamstar to be used for smoothing forecasts when being
        estimated empirically.

    Returns
    -------
    df2 : pd.Dataframe
        Output dataframe with all reconciled forecasts filled into the original
        input. 


    """
    def __init__(self,
                 df: pd.DataFrame,
                 forecaster = DefaultForecaster(),
                 equality_constraints :list[str] = [],
                 inequality_constraints :list[str] = [],
                 parallelize:bool = True,
                 n_sample_split:int = 5,
                 shrinkage_method:str = 'oas',
                 lamstar_empirically:bool = True,
                 default_lam:float = 6.25,
                 max_lam:float = 129600):
        
        self.df = df
        self.forecaster = forecaster
        self.equality_constraints = equality_constraints
        self.inequality_constraints = inequality_constraints
        self.parallelize = parallelize
        self.n_sample_split = n_sample_split
        self.shrinkage_method = shrinkage_method
        self.lamstar_empirically = lamstar_empirically
        self.default_lam = default_lam
        self.max_lam = max_lam

        
    def fit(self):
        """
        Fits the model and generates reconciled forecasts for the input 
        dataframe subject to defined constraints.
        """

        df = self.df
        forecaster = self.forecaster
        equality_constraints = self.equality_constraints
        inequality_constraints = self.inequality_constraints
        parallelize = self.parallelize
        n_sample_split = self.n_sample_split 
        shrinkage_method = self.shrinkage_method
        lamstar_empirically = self.lamstar_empirically
        default_lam = self.default_lam
        max_lam = self.max_lam
        
        # modify inputs into machine-friendly shape
        df0, all_cells, unknown_cells, known_cells, islands = OrganizeCells(df)
        C,d = StringToMatrixConstraints(df0.T.stack(),
                                        all_cells,
                                        unknown_cells,
                                        known_cells,
                                        equality_constraints)
        C,d = AddIslandsToConstraints(C,d,islands)
        C_ineq,d_ineq = StringToMatrixConstraints(df0.T.stack(),
                                                  all_cells,
                                                  unknown_cells,
                                                  known_cells,
                                                  inequality_constraints)
        # 1st stage forecast and its model
        df1,df1_model = FillAllEmptyCells(df0,forecaster,parallelize=parallelize)

        # get pseudo out-of-sample prediction, true values, and prediction models
        pred,true,model = GenPredTrueData(df0,forecaster,n_sample=n_sample_split, 
                                          parallelize=parallelize)
        
        # break dataframe into list of time series
        ts_list,pred_list,true_list = BreakDataFrameIntoTimeSeriesList(df0,df1,pred,true)
        
        # get parts for reconciliation
        y1 = GenVecForecastWithIslands(ts_list,islands)
        W,shrinkage = GenWeightMatrix(pred_list, true_list,
                                      method = shrinkage_method)
        smoothness = GenLamstar(pred_list,true_list,
                                empirically = lamstar_empirically,
                                default_lam = default_lam,
                                max_lam = max_lam)
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
