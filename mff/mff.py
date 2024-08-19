from .step0_parse_constraints import generate_constraints
from .step2_reconciler import Reconciler
from .step1_unreconciled_forecast import UnreconciledForecaster
from .default_forecaster import get_default_forecaster

import warnings

class MFF:  # TODO: this can probably be done more smartly using inheritance
    """
       Macroframework forecaster (MFF) class for generating and reconciling forecasts.

       Attributes:
           df (DataFrame): DataFrame containing historical data and exogenous values. Rows with NAs correspond to horizons that should be forecasted.
           constraints_list (list): List of string constraints to be parsed for reconciliation.
           lam (float): Smoothing parameter based on HP filter smoothing. Default is 100.
           forecaster (object): sktime object used for generating forecasts. Default is the result of get_default_forecaster(1).
           n_resid (int): Number of residuals to be used for covariance calculation. Default is 5.
           cov_calc_method (str): Method for covariance calculation. Default is 'oasd'.
           ignore_warnings (bool): Flag to ignore warnings during unconditional forecast estimation. Default is True.
           unconditional_forecaster (UnreconciledForecaster): UnreconciledForecaster object for generating unconditional forecasts for unknown variables.
           reconciler (Reconciler): Reconciler object that takes in unreconciled forecasts and constraints to returns reconciled forecasts.
           y_reconciled (array): Reconciled forecast values.
           y_unreconciled (array): Unreconciled forecast values.
           C (array): Matrix containing constraint coefficents.
           d (array): Vector containing constants for constraints.
           W (array): Covariance matrix of unknown variable forecast errors.
           fitted_forecasters (list): List of fitted forecasters. Can only be used with a forecater that is of ForecastingGridSearchCV class
       """

    def __init__(self,
                 df,
                 constraints_list,
                 lam=100,
                 forecaster=get_default_forecaster(1),
                 n_resid=5,
                 cov_calc_method='oasd',
                 ignore_forecaster_warnings=True
                 ):
        """
        Initializes the MFF class with the given parameters.

        Args:
            df (DataFrame): DataFrame containing historical data and exogenous values. Rows with NAs correspond to horizons that should be forecasted.
            constraints_list (list): List of string constraints to be parsed for reconciliation.
            lam (float): Smoothing parameter based on HP filter smoothing. Default is 100.
            forecaster (object): sktime object used for generating forecasts. Default is the result of get_default_forecaster(1).
            n_resid (int): Number of residuals to be used for covariance calculation. Default is 5.
            cov_calc_method (str): Method for covariance calculation. Default is 'oasd'.
            ignore_forecaster_warnings (bool): Flag to ignore warnings during unconditional forecast estimation. Default is True.
        """
        self.df = df
        self.constraints_list = constraints_list
        self.forecaster = forecaster
        self.lam = lam
        self.n_resid = n_resid
        self.cov_calc_method = cov_calc_method
        self.ignore_warnings = ignore_forecaster_warnings

        self.reconciler = None
        self.unconditional_forecaster = None

        self.y_reconciled = None
        self.y_unreconciled = None
        self.C = None
        self.d = None
        self.W = None
        self.fitted_forecasters = None

    def parse_constraints(self):
        """
        Parses the constraints and generates the constraint matrix (C) and vector (d).
        """
        self.C, self.d = generate_constraints(self.df, self.constraints_list)

    def fit_unconditional_forecaster(self):
        """
        Fits the unconditional forecaster and calculates the covariance matrix using the method in self.cov_calc_method.
        """
        self.unconditional_forecaster = UnreconciledForecaster(self.df, self.forecaster)
        if self.ignore_warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.unconditional_forecaster.fit()
                self.unconditional_forecaster.fit_covariance(self.n_resid, self.cov_calc_method)
        else:
            self.unconditional_forecaster.fit()
            self.unconditional_forecaster.fit_covariance(self.n_resid, self.cov_calc_method)
        self.y_unreconciled = self.unconditional_forecaster.y_hat
        self.W = self.unconditional_forecaster.cov.cov_mat

    def fit_reconciler(self):
        """
        Fits the reconciler using the unconditional forecasts from self.unconditional_foreacster and constraints
        self.C and self.d.
        """
        self.reconciler = Reconciler(self.unconditional_forecaster.y_hat,
                                     self.unconditional_forecaster.df,
                                     self.unconditional_forecaster.cov.cov_mat,
                                     self.C,
                                     self.d,
                                     self.lam
                                     )
        self.reconciler.fit()
        self.y_reconciled = self.reconciler.y_reconciled

    def fit(self):
        """
        Fits the entire MFF model by generating constraint matrices, fitting the unconditional forecaster, and reconciling forecasts.
        """
        print('Generating constraints')
        self.parse_constraints()

        print('Generating forecasts')
        self.fit_unconditional_forecaster()

        print('Reconciling forecasts')
        self.fit_reconciler()
