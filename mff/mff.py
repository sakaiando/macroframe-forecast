from .step0_parse_constraints import generate_constraints
from .step2_reconciler import Reconciler
from .step1_unreconciled_forecast import UnreconciledForecaster
from .default_forecaster import get_default_forecaster

import warnings

class MFF:  # TODO: this can probably be done more smartly using inheritance
    def __init__(self,
                 df,
                 constraints_list,
                 lam=100,
                 forecaster=get_default_forecaster(1),
                 n_resid=5,
                 cov_calc_method='oasd',
                 ignore_forecaster_warnings=True
                 ):
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
        self.C, self.d = generate_constraints(self.df, self.constraints_list)

    def fit_unconditional_forecaster(self):
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
        print('Generating constraints')
        self.parse_constraints()

        print('Generating forecasts')
        self.fit_unconditional_forecaster()

        print('Reconciling forecasts')
        self.fit_reconciler()
