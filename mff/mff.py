from .constraint_parser import generate_constraints
from .step2_reconciler import Reconciler
from .step1_forecast import UnconditionalForecaster


class MFF:  # TODO: this can probably be done more smartly using inheritance
    def __init__(self, df, constraints_list, forecaster, lam, n_resid=5, cov_calc_method='oasd'):
        self.df = df
        self.constraints_list = constraints_list
        self.forecaster = forecaster
        self.lam = lam
        self.n_resid = n_resid
        self.cov_calc_method = cov_calc_method

        self.y_hat = None
        self.reconciler = None
        self.unconditional_forecaster = None
        self.C = None
        self.b = None

    def generate_constraints(self):
        self.C, self.b = generate_constraints(self.df, self.constraints_list)

    def fit_unconditional_forecaster(self):
        self.unconditional_forecaster = UnconditionalForecaster(self.df, self.forecaster)
        self.unconditional_forecaster.fit_covariance(self.n_resid, self.cov_calc_method)
        self.unconditional_forecaster.fit()

    def fit_reconciler(self):
        self.reconciler = Reconciler(self.unconditional_forecaster.y_hat,
                                     self.unconditional_forecaster.df,
                                     self.unconditional_forecaster.cov.cov_mat,
                                     self.C,
                                     self.b,
                                     self.lam
                                     )
        self.reconciler.fit()
        self.y_hat = self.reconciler.y_reconciled

    def fit(self):
        print('Generating constraints')
        self.generate_constraints()

        print('Generating forecasts')
        self.fit_unconditional_forecaster()

        print('Reconciling forecasts')
        self.fit_reconciler()
