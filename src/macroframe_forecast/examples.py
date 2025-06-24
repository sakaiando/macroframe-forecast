# Disclaimer: Reuse of this tool and IMF information does not imply
# any endorsement  of the research and/or product. Any research presented
# should not be reported as representing the views of the IMF,
# its Executive Board, member governments.


from string import ascii_uppercase

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
