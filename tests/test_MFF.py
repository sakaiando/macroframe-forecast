# Disclaimer: Reuse of this tool and IMF information does not imply
# any endorsement  of the research and/or product. Any research presented
# should not be reported as representing the views of the IMF,
# its Executive Board, member governments.


from string import ascii_uppercase
from typing import cast

import numpy as np
import pandas as pd
from pytest import mark

from macroframe_forecast import MFF, MFF_mixed_freqency
from macroframe_forecast.utils import AddIslandsToConstraints

# %%


def test_add_islands_to_constraints_with_mixed_index_types():
    island_index = pd.MultiIndex.from_tuples([("B0", 2000)])
    cell_index = pd.MultiIndex.from_tuples([("A0", 2001), ("B0", 2000)])
    constraints = pd.DataFrame([[1.0, -1.0]], index=["A0_2001-B0_2000"], columns=cell_index)
    rhs = pd.DataFrame([0.0], index=constraints.index)
    islands = pd.Series([2.0], index=island_index)

    augmented, augmented_rhs = AddIslandsToConstraints(constraints, rhs, islands)

    assert list(augmented.index) == [("B0", 2000), "A0_2001-B0_2000"]
    assert augmented.at[("B0", 2000), ("B0", 2000)] == 1
    assert augmented_rhs.at[("B0", 2000), 0] == 2
    assert augmented.loc["A0_2001-B0_2000"].equals(cast(pd.Series, constraints.loc["A0_2001-B0_2000"]))


def test_mixed_frequency_reconciles_annual_quarterly_identity():
    from sktime.forecasting.naive import NaiveForecaster

    quarters = pd.period_range("2000Q1", periods=48, freq="Q")
    quarterly_true = pd.DataFrame({"a": np.arange(1, 49, dtype=float) ** 2}, index=quarters)
    annual_true = quarterly_true.groupby(cast(pd.PeriodIndex, quarterly_true.index).asfreq("Y")).sum()
    quarterly = quarterly_true.copy()
    annual = annual_true.copy()
    quarterly.iloc[-8:] = np.nan
    annual.iloc[-3:] = np.nan
    model = MFF_mixed_freqency(
        {"Y": annual, "Q": quarterly},
        forecaster=NaiveForecaster(strategy="mean"),
        constraints_with_wildcard=["?Q1+?Q2+?Q3+?Q4-?"],
    )

    annual_result, quarterly_result = model.fit()

    assert annual_result.index.equals(annual.index)
    assert quarterly_result.index.equals(quarterly.index)
    assert annual_result.columns.equals(annual.columns)
    assert quarterly_result.columns.equals(quarterly.columns)
    assert annual_result.notna().all().all()
    assert quarterly_result.notna().all().all()
    pd.testing.assert_frame_equal(annual_result.iloc[:-3], annual.iloc[:-3])
    pd.testing.assert_frame_equal(quarterly_result.iloc[:-8], quarterly.iloc[:-8])
    quarterly_totals = quarterly_result.groupby(cast(pd.PeriodIndex, quarterly_result.index).asfreq("Y")).sum()
    np.testing.assert_allclose(annual_result, quarterly_totals, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(annual_result.iloc[-3], annual_true.iloc[-3], rtol=1e-8, atol=1e-8)


@mark.slow
def test_MFF_non_parallel():
    n = 30
    p = 3
    fh = 1
    df_true = pd.DataFrame(
        np.random.rand(n, p),
        columns=[f"{L}{i}" for i in range(int(np.ceil(p / 26))) for L in ascii_uppercase][:p],
        index=pd.date_range(start="2000", periods=n, freq="YE").year,
    )
    df_true.iloc[:, -1] = df_true.iloc[:, :-1].sum(axis=1)
    df = df_true.copy()
    df.iloc[-fh:, : np.ceil(p / 2).astype(int)] = np.nan
    df.iloc[-1, 0] = df_true.iloc[-1, 0]  # island
    equality_constraints = ["A0?+B0?-C0?"]

    m = MFF(df, equality_constraints=equality_constraints, parallelize=False)
    df2 = m.fit()

    assert df2.iloc[-1, 0] == df_true.iloc[-1, 0]


@mark.slow
def test_MFF_parallel():
    n = 30
    p = 3
    fh = 1
    df_true = pd.DataFrame(
        np.random.rand(n, p),
        columns=[f"{L}{i}" for i in range(int(np.ceil(p / 26))) for L in ascii_uppercase][:p],
        index=pd.date_range(start="2000", periods=n, freq="YE").year,
    )
    df_true.iloc[:, -1] = df_true.iloc[:, :-1].sum(axis=1)
    df = df_true.copy()
    df.iloc[-fh:, : np.ceil(p / 2).astype(int)] = np.nan
    df.iloc[-1, 0] = df_true.iloc[-1, 0]  # island

    equality_constraints = ["A0?+B0?-C0?"]

    m = MFF(df, equality_constraints=equality_constraints, parallelize=True)
    df2 = m.fit()

    assert df2.iloc[-1, 0] == df_true.iloc[-1, 0]


@mark.slow
def test_MFF_mixed_frequency():
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
    dfA_true = dfQ_true.groupby(cast(pd.PeriodIndex, dfQ_true.index).year).sum()
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
    assert len(df2_list) == 2
    assert df2_list[0].index.equals(dfA.index)
    assert df2_list[1].index.equals(dfQ.index)
    assert df2_list[0].iloc[-fhA:, :2].notna().all().all()
    assert df2_list[1].iloc[-fhQ:, :2].notna().all().all()
    assert ~np.isnan(cast(float, df2_list[0].iloc[-1, 0]))


@mark.slow
def test_small_sample_MFF():
    n = 20
    p = 2
    fh = 5
    df_true = pd.DataFrame(
        np.random.rand(n, p),
        columns=[f"{L}{i}" for i in range(int(np.ceil(p / 26))) for L in ascii_uppercase][:p],
        index=pd.date_range(start="2000", periods=n, freq="YE").year,
    )
    # df_true.iloc[:,-1] = df_true.iloc[:,:-1].sum(axis=1)
    df = df_true.copy()
    df.iloc[-fh:, : np.ceil(p / 2).astype(int)] = np.nan
    # df.iloc[-1,0] = df_true.iloc[-1,0] # island
    equality_constraints = []

    m = MFF(df, equality_constraints=equality_constraints, parallelize=False)
    df2 = m.fit()

    assert ~np.isnan(cast(float, df2.iloc[-1, 0]))


@mark.slow
def test_inequality_constraints():

    n = 20
    p = 2
    fh = 5
    df_true = pd.DataFrame(
        np.random.rand(n, p),
        columns=[f"{L}{i}" for i in range(int(np.ceil(p / 26))) for L in ascii_uppercase][:p],
        index=pd.date_range(start="2000", periods=n, freq="YE").year,
    )
    # df_true.iloc[:,-1] = df_true.iloc[:,:-1].sum(axis=1)
    df = df_true.copy()
    df.iloc[-fh:, : np.ceil(p / 2).astype(int)] = np.nan
    # df.iloc[-1,0] = df_true.iloc[-1,0] # island

    equality_constraints = []

    inequality_constraints = [ df.columns[0] + '_' + str(df_true.index[-1]) + ' + 1']

    m = MFF(df, equality_constraints=equality_constraints, 
            inequality_constraints = inequality_constraints, 
            parallelize=False)
    df2 = m.fit()
    df2.iloc[-1, 0]

    assert (cast(float, df2.iloc[-1, 0]) <= -1) or np.isclose(cast(float, df2.iloc[-1, 0]), -1, atol=1e-12)


@mark.slow
def test_equality_constraints():

    n = 20
    p = 2
    fh = 5
    df_true = pd.DataFrame(
        np.random.rand(n, p),
        columns=[f"{L}{i}" for i in range(int(np.ceil(p / 26))) for L in ascii_uppercase][:p],
        index=pd.date_range(start="2000", periods=n, freq="YE").year,
    )
    # df_true.iloc[:,-1] = df_true.iloc[:,:-1].sum(axis=1)
    df = df_true.copy()
    df.iloc[-fh:, : np.ceil(p / 2).astype(int)] = np.nan
    # df.iloc[-1,0] = df_true.iloc[-1,0] # island

    equality_constraints = [df.columns[0] + '_' + str(df_true.index[-1]) + ' + 1']

    inequality_constraints = []

    m = MFF(df, equality_constraints=equality_constraints, 
            inequality_constraints = inequality_constraints, 
            parallelize=False)
    df2 = m.fit()
    df2.iloc[-1, 0]

    assert round(cast(float, df2.iloc[-1, 0]), 2) == -1
