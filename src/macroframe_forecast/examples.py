# Disclaimer: Reuse of this tool and IMF information does not imply
# any endorsement  of the research and/or product. Any research presented
# should not be reported as representing the views of the IMF,
# its Executive Board, member governments.


from string import ascii_uppercase
from pandas import DataFrame
import numpy as np
import pandas as pd
from sktime.datasets import load_macroeconomic

from macroframe_forecast import MFF, MFF_mixed_freqency

# %%


def example1():  # no constraints
    # load data
    # from sktime.datasets import load_macroeconomic
    df_true = load_macroeconomic().iloc[:, :5]

    # input dataframe
    df = df_true.copy()
    fh = 5
    df.iloc[-fh:, 0] = np.nan

    # apply MFF
    m = MFF(df, equality_constraints=[])
    df2 = m.fit()
    df0 = m.df0
    df1 = m.df1
    df1_model = m.df1_model
    smoothness = m.smoothness
    shrinkage = m.shrinkage

    # plot results
    t0 = -30
    ax = df0.iloc[t0:, 0].plot(label="df0")
    df1.iloc[t0:, 0].plot(ax=ax, label="df1")
    df2.iloc[t0:, 0].plot(ax=ax, label="df2")
    df_true.iloc[t0:, 0].plot(ax=ax, label="df_true")
    ax.axvline(x=df0.index[-fh])
    ax.legend()

    print("smoothness", smoothness.values)
    print("shrinkage", np.round(shrinkage, 3))
    for ri, ci in np.argwhere(df.isna()):
        print(df1_model.index[ri], df1_model.columns[ci], df1_model.iloc[ri, ci].best_params_)


# example 2: with constraints
def example2():
    # create data
    n = 30
    p = 3
    fh = 5
    df_true = pd.DataFrame(
        np.random.rand(n, p),
        columns=[f"{L}{i}" for i in range(int(np.ceil(p / 26))) for L in ascii_uppercase][:p],
        index=pd.date_range(start="2000", periods=n, freq="YE").year,
    )
    df_true.iloc[:, -1] = df_true.iloc[:, :-1].sum(axis=1)
    df = df_true.copy()
    df.iloc[-fh:, : np.ceil(p / 2).astype(int)] = np.nan
    df.iloc[-1, 0] = df_true.iloc[-1, 0]  # island
    # df.iloc[-fh,-1] = df.iloc[:,-1].mean()
    # df.iloc[-3,1] = df_true.iloc[-3,1] # island
    equality_constraints = ["A0?+B0?-C0?"]
    # ineq_constraints_with_wildcard = ['A0?-0.5'] # A0 <=0.5 for all years

    # fit data
    m = MFF(df, equality_constraints=equality_constraints)
    df2 = m.fit()
    df0 = m.df0
    df1 = m.df1
    df1_model = m.df1_model
    shrinkage = m.shrinkage
    smoothness = m.smoothness
    # TODO: delete, the assignment below, if not needed
    W = m.W  # noqa: F841
    for ri, ci in np.argwhere(df.isna()):
        print(df1_model.index[ri], df1_model.columns[ci], df1_model.iloc[ri, ci].best_params_)

    import matplotlib.pyplot as plt

    plt.figure()
    t0 = -20
    plt.subplot(2, 1, 1)
    ax = df0.iloc[t0:, 0].plot(label="df0")
    df1.iloc[t0:, 0].plot(ax=ax, label="df1")
    df2.iloc[t0:, 0].plot(ax=ax, label="df2")
    df_true.iloc[t0:, 0].plot(ax=ax, label="df_true")
    ax.axvline(x=df0.index[-fh])

    plt.subplot(2, 1, 2)
    ax = df0.iloc[t0:, 1].plot(label="df0")
    df1.iloc[t0:, 1].plot(ax=ax, label="df1")
    df2.iloc[t0:, 1].plot(ax=ax, label="df2")
    df_true.iloc[t0:, 1].plot(ax=ax, label="df_true")
    ax.axvline(x=df0.index[-fh], label="fh=1")
    ax.legend(loc="lower left")

    print("smoothness", smoothness.values)
    print("shrinkage", np.round(shrinkage, 3))

    # confirm constraints
    assert np.isclose(df2["A0"] + df2["B0"] - df2["C0"], 0).all()


# example, mixed-frequency intra-inter-temporal constraints
def example3():
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    n = 120
    p = 3
    fhA = 5
    fhQ = 7
    dfQ_true = pd.DataFrame(
        np.random.rand(n, p),
        columns=[f"{L}{i}" for i in range(int(np.ceil(p / 26))) for L in ascii_uppercase][:p],
        index=pd.period_range(start="2000-1-1", periods=n, freq="Q"),
    )
    dfQ_true.iloc[:, -1] = dfQ_true.iloc[:, :-1].sum(axis=1)
    dfA_true = dfQ_true.groupby(dfQ_true.index.year).sum()
    dfA_true.index = pd.PeriodIndex(dfA_true.index, freq="Y")

    dfA = dfA_true.copy()
    dfA.iloc[-fhA:, : np.ceil(p / 2).astype(int)] = np.nan

    dfQ = dfQ_true.iloc[:-12, :].copy()
    dfQ.iloc[-fhQ:, : np.ceil(p / 2).astype(int)] = np.nan

    # inputs
    df_dict = {"Y": dfA, "Q": dfQ}
    constraints_with_wildcard = ["A0?+B0?-C0?", "?Q1+?Q2+?Q3+?Q4-?"]

    mff = MFF_mixed_freqency(df_dict, constraints_with_wildcard=constraints_with_wildcard)
    df2_list = mff.fit()
    df1_list = mff.df1_list
    df0_list = mff.df0_list

    # plot results
    import matplotlib.pyplot as plt

    t0 = -30
    plt.subplot(2, 1, 1)
    ax = df0_list[1].iloc[t0:, 0].plot(label="df0")
    df1_list[1].iloc[t0:, 0].plot(ax=ax, label="df1")
    df2_list[1].iloc[t0:, 0].plot(ax=ax, label="df2")
    dfQ_true.iloc[t0:, 0].plot(ax=ax, label="df_true")
    ax.axvline(x=df0_list[1].index[-fhQ], label="fh=1")
    ax.legend(loc="lower left")

    plt.subplot(2, 1, 2)
    ax = df0_list[0].iloc[t0:, 0].plot(label="df0")
    df1_list[0].iloc[t0:, 0].plot(ax=ax, label="df1")
    df2_list[0].iloc[t0:, 0].plot(ax=ax, label="df2")
    dfA_true.iloc[t0:, 0].plot(ax=ax, label="df_true")
    ax.axvline(x=df0_list[0].index[-fhQ], label="fh=1")
    ax.legend(loc="lower left")

    # check constraints
    df2A = df2_list[0]
    df2Q = df2_list[1]
    df2A.eval("A0+B0-C0")
    (df2Q.resample("Y").sum() - df2A).dropna()


def generate_example_GDP_df() -> DataFrame:
    """Utility function to generate example GDP data for quick demonstration purposes.

    Example:

    ```python
    from macroframe_forecast import MFF
    from macroframe_forecast.examples import generate_example_GDP_df

    df0 = generate_example_GDP_df()
    m = MFF(df0, equality_constraints=["GDP_2030 - 1.04 * GDP_2029"])
    m.fit()
    ```

    """
    GDP_data_true = DataFrame(
        {
            "year": [
                1950,
                1951,
                1952,
                1953,
                1954,
                1955,
                1956,
                1957,
                1958,
                1959,
                1960,
                1961,
                1962,
                1963,
                1964,
                1965,
                1966,
                1967,
                1968,
                1969,
                1970,
                1971,
                1972,
                1973,
                1974,
                1975,
                1976,
                1977,
                1978,
                1979,
                1980,
                1981,
                1982,
                1983,
                1984,
                1985,
                1986,
                1987,
                1988,
                1989,
                1990,
                1991,
                1992,
                1993,
                1994,
                1995,
                1996,
                1997,
                1998,
                1999,
                2000,
                2001,
                2002,
                2003,
                2004,
                2005,
                2006,
                2007,
                2008,
                2009,
                2010,
                2011,
                2012,
                2013,
                2014,
                2015,
                2016,
                2017,
                2018,
                2019,
                2020,
                2021,
                2022,
                2023,
                2024,
                2025,
                2026,
                2027,
                2028,
                2029,
                2030,
            ],
            "GDP": [
                301782704906.154,
                348993057004.926,
                368027835977.609,
                389147698401.843,
                390276672099.46,
                424868331217.657,
                448388356231.708,
                471707274214.225,
                478166880805.205,
                519476064642.104,
                539899866168.654,
                558583293630.287,
                600454646133.34,
                633368190949.311,
                680153540812.135,
                737201978910.734,
                808045440847.441,
                853883822469.0601,
                933096436159.1281,
                1008751520510.61,
                1064366709379.28,
                1155403629216.3,
                1269884411457.22,
                1418456050381.57,
                1536647924378.57,
                1674009506825.93,
                1867242215504.46,
                2079644632633.34,
                2350400768409.49,
                2627325000000.0,
                2857325000000.0,
                3207025000000.0,
                3343800000000.0,
                3634025000000.0,
                4037650000000.0,
                4339000000000.0,
                4579625000000.0,
                4855250000000.0,
                5236425000000.0,
                5641600000000.0,
                5963125000000.0,
                6158125000000.0,
                6520325000000.0,
                6858550000000.0,
                7287250000000.0,
                7639750000000.0,
                8073125000000.0,
                8577550000000.0,
                9062825000000.0,
                9631175000000.0,
                10250950000000.0,
                10581925000000.0,
                10929100000000.0,
                11456450000000.0,
                12217175000000.0,
                13039200000000.0,
                13815600000000.0,
                14474250000000.0,
                14769850000000.0,
                14478050000000.0,
                15048975000000.0,
                15599725000000.0,
                16253950000000.0,
                16880675000000.0,
                17608125000000.0,
                18295000000000.0,
                18804900000000.0,
                19612100000000.0,
                20656525000000.0,
                21539975000000.0,
                21354125000000.0,
                23681175000000.0,
                26006900000000.0,
                27720725000000.0,
                29184900000000.0,
                30507217002511.25,
                31717641479090.75,
                32941710359665.25,
                34342131994149.0,
                35712823521822.0,
                37153089058192.75,
            ],
        }
    )

    # The original GDP data is in dollar numbers, but changing this to billions
    # going forward in order to deal with problem of matrix invertibility.
    GDP_data_true["GDP"] = GDP_data_true["GDP"] / 1e12

    # Time period hs to be set as the index. Here year is the time identifier,
    # therefore setting this as the index.
    GDP_data_true.set_index(GDP_data_true["year"], inplace=True)
    GDP_data_true.drop(columns="year", inplace=True)

    # Creating a copy which is used for geenrating the forecasts. Removing the last
    # six years of data for ease of forecasts
    GDP_data = GDP_data_true.copy()
    # Removing the last six years of data so that they are forecasted by the
    # function.
    GDP_data.iloc[-6:, 0] = np.nan
    return GDP_data
