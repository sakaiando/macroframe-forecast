import imf_datatools.ecos_sdmx_utilities as ecos
import numpy as np
import pandas as pd
import re
from joblib import Parallel, delayed
from tqdm import tqdm
from mff.mff import MFF
from mff.ecos_reader import process_ecos_df
import imf_datatools.edi_utilities as edi


import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

db_list = ecos.get_databases()

pattern = r"WEO_WEO([A-Za-z]{3}\d{4})Pub$"
db_names = db_list.index.to_frame()['dbname'].str.extract(pattern).dropna()
db_names.columns = ['date']
db_names['date'] = pd.PeriodIndex(db_names['date'], freq='M')
db_names['year'] = pd.PeriodIndex(db_names['date'], freq='M').year
db_names = db_names.sort_values(by='date')

endog_vars = ['BIP_GDP_BP6',
              'BIS_GDP_BP6',
              'BCA_GDP_BP6',
              'BGS_GDP_BP6',
              ]
# BFO_BP6
exog_vars = ['NGDP_RPCH',
             'NGDP_RPCH111',
             'NGDP_RPCH134',
             ]

constraints = ['BIS_GDP_BP6_? + BIP_GDP_BP6_? + BGS_GDP_BP6_? - BCA_GDP_BP6_?']

country = '714'

# get most recent vintage, keep historical data
pub_actual = db_names.index[-1]

class WEOdf:
    def __init__(self, pub, country, endog_vars, freq, exog_vars=None, longformat=True):
        self.pub = pub
        self.country = country
        self.endog_vars = endog_vars
        self.exog_vars = exog_vars if exog_vars is not None else []
        self.vars = self.endog_vars + self.exog_vars
        self.freq = freq
        self.longformat = longformat

        self.process_attributes()

    def process_attributes(self):
        self.get_weo_date()
        self.load_df()
        self.split_hist_fcast()

    def get_weo_date(self):
        self.pub_date = pd.Period(re.match(pattern, self.pub).group(1))
        self.pub_year = self.pub_date.year
        self.pub_month = self.pub_date.month
        self.hist_end = self.pub_year - 1

    def load_df(self):
        df = ecos.get_ecos_sdmx_data(self.pub, self.country, self.vars, freq='A', longformat=self.longformat)
        if df is None:
            self.df = df
        else:
            df.columns = df.columns.str.split('.', n=1, expand=True).droplevel(1)
            if self.longformat:
                df = df.set_index('dates').drop('COUNTRY', axis=1)
            else:
                df.columns = [col[3:] + col[:3] for col in df.columns]
            df.index = pd.PeriodIndex(df.index, freq='A')
            df.columns.name = 'variable'
            self.df = df

    def split_hist_fcast(self):
        if self.df is not None:
            self.df_hist = self.df.loc[:str(self.hist_end)].dropna()
            self.df_fcast = self.df.loc[str(self.pub_year):]
            self.df_with_gaps = self.df.copy().dropna()
            self.df_with_gaps.loc[str(self.pub_year):, endog_vars] = np.nan
        else:
            self.df_hist = None
            self.df_fcast = None
            self.df_with_gaps = None

    def append_columns(self, df):
        self.df = pd.concat([self.df, df])
        self.split_hist_fcast()

weo_actual = WEOdf(pub_actual, country, endog_vars, freq='A', exog_vars=exog_vars)
weo_exog = WEOdf(pub_actual, ['111', '134'], [], freq='A', exog_vars=['NGDP_RPCH'], longformat=False)

df_hist = weo_actual.df_hist
df_hist.index.name = 'year'
df_hist.index = df_hist.index.year
df_hist_for_mff = process_ecos_df(weo_actual.df_hist).unstack(['freq', 'subperiod'])

def process_for_mff(weo_actual, weo_exog):
    df = weo_actual.df_with_gaps.copy()
    df.index.name = 'year'
    df.index = df.index.year

    exog_df = weo_exog.df_with_gaps.dropna(axis=1).copy()
    exog_df.index.name = 'year'
    exog_df.index = exog_df.index.year

    df = df.join(exog_df)
    df = process_ecos_df(df).unstack(['freq', 'subperiod'])
    return df


def make_resids(df_actual, df_fcast):
    err = (df_fcast - df_actual).dropna(axis=1, how='all').dropna(axis=0, how='all')
    err.index = err.index - weo_curr.pub_year
    err.index.name = 'horizon'
    return err


# loop over past vintages, get forecasts
errs = []
errs_mff_reconciled = []
errs_mff_unreconciled = []
for pub in tqdm(db_names.index[62:]):
    weo_curr = WEOdf(pub, country, endog_vars, freq='A', exog_vars=exog_vars, longformat=True)
    weo_exog = WEOdf(pub, ['111', '134'], [], freq='A', exog_vars=['NGDP_RPCH'], longformat=False)

    if weo_curr.df is not None:
        # MFF error
        df = process_for_mff(weo_curr, weo_exog)
        if len(df.dropna()) == 0:
            continue
        mff = MFF(df, constraints, n_resid=2)
        mff.fit()

        err_mff = make_resids(mff.y_reconciled, df_hist)
        err_mff_unrec = make_resids(mff.y_unreconciled, df_hist)

        df_weo_curr = weo_curr.df_fcast
        df_weo_curr.index.name = 'year'
        df_weo_curr.index = df_weo_curr.index.year
        df_weo_actual = weo_actual.df_hist
        df_weo_actual.index.name = 'year'

        err = make_resids(weo_curr.df_fcast, weo_actual.df_hist)

        errs_mff_unreconciled.append(err_mff_unrec)
        errs_mff_reconciled.append(err_mff)
        errs.append(err)

df_errs = pd.concat(errs).stack()
df_errs.name = 'err'
df_errs = df_errs.to_frame()

df_errs['sq_err'] = df_errs['err'] ** 2
df_errs_grp = df_errs.groupby(['variable', 'horizon']).mean()
df_errs_grp['sq_err'] = df_errs_grp['sq_err'] ** .5

df_errs['err'].unstack(['horizon', 'variable']).cov()
