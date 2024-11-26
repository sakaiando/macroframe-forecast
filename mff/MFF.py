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
