import imf_datatools.ecos_sdmx_utilities as ecos
import pandas as pd
from tqdm import tqdm
db_list = ecos.get_databases()

pattern = r"WEO_WEO([A-Za-z]{3}\d{4})Pub$"
db_names = db_list.index.to_frame()['dbname'].str.extract(pattern).dropna()
db_names.columns = ['date']
db_names['date'] = pd.PeriodIndex(db_names['date'], freq='M')
db_names['year'] = pd.PeriodIndex(db_names['date'], freq='M').year
db_names = db_names.sort_values(by='date')



vars = ['BIP_GDP_BP6', 'BIS_GDP_BP6', 'NGDP_RPCH']
country = '714'


#get most recent vintage, keep historical data
pub_actual = db_names.index[-1]
df_actual = ecos.get_ecos_sdmx_data(pub_actual, country, vars, freq='A', longformat=True)
year = db_names.loc[pub_actual, 'year']
df_actual = df_actual.loc[df_actual['dates'] <= str(year)]
df_actual.columns = df_actual.columns.str.split('.', n=1, expand=True).droplevel(1)
df_actual = df_actual.set_index('dates').drop('COUNTRY', axis=1)


# loop over past vintages, get forecasts
errs = []
for pub in tqdm(db_names.index):
    df = ecos.get_ecos_sdmx_data(pub, country, vars, freq='A', longformat=True)
    if not df is None:
        df.columns = df.columns.str.split('.', n=1, expand=True).droplevel(1)
        year = db_names.loc[pub, 'year']
        df_fcast = df.loc[df['dates'] >= str(year)]
        df_fcast = df_fcast.set_index('dates').drop('COUNTRY', axis=1)

        err = (df_actual - df_fcast).dropna()
        if len(err) > 0:
            err.index = err.index.year - year
            err['pub'] = db_names.loc[pub, 'date']
            err.index.name = 'horizon'
            err = err.set_index('pub', append=True)
            err.columns.name = 'variable'
            errs.append(err)

df_errs = pd.concat(errs).stack()
df_errs.name = 'err'
df_errs = df_errs.to_frame()

df_errs['sq_err'] = df_errs['err'] ** 2
df_errs_grp = df_errs.groupby(['variable', 'horizon']).mean()
df_errs_grp['sq_err'] = df_errs_grp['sq_err'] ** .5


df_errs['err'].unstack(['horizon', 'variable']).cov()