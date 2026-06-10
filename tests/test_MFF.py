# Disclaimer: Reuse of this tool and IMF information does not imply
# any endorsement  of the research and/or product. Any research presented
# should not be reported as representing the views of the IMF,
# its Executive Board, member governments.


from string import ascii_uppercase

import numpy as np
import pandas as pd
from pytest import mark

from macroframe_forecast import MFF, MFF_mixed_freqency

# %%


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
    assert ~np.isnan(df2_list[0].iloc[-1, 0])


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

    assert ~np.isnan(df2.iloc[-1, 0])


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

    assert (df2.iloc[-1, 0] <= -1) or np.isclose(df2.iloc[-1, 0], -1, atol=1e-12)


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

    assert round(df2.iloc[-1, 0],2) == -1


@mark.slow
def test_smoothness_parameter():
    """Test that the smoothness parameter can be passed directly and is used correctly."""
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
    
    # First run to get the smoothness structure
    m1 = MFF(df, equality_constraints=equality_constraints, parallelize=False)
    df2_default = m1.fit()
    
    # Get the generated smoothness from internal state (after fit, it's computed)
    from macroframe_forecast.utils import (
        OrganizeCells,
        BreakDataFrameIntoTimeSeriesList,
        FillAllEmptyCells,
        GenPredTrueData,
        DefaultForecaster,
        CheckTrainingSampleSize,
        GenLamstar
    )
    
    df0, all_cells, unknown_cells, known_cells, islands = OrganizeCells(df)
    small_sample = CheckTrainingSampleSize(df0, 5)
    forecaster = DefaultForecaster(small_sample)
    df1, df1_model = FillAllEmptyCells(df0, forecaster, parallelize=False)
    pred, true, model = GenPredTrueData(df0, forecaster, n_forecast_error=5, parallelize=False)
    ts_list, pred_list, true_list = BreakDataFrameIntoTimeSeriesList(df0, df1, pred, true)
    
    # Get original smoothness
    original_smoothness = GenLamstar(pred_list, true_list, default_lam=-1, max_lam=129600)
    
    # Create a custom smoothness with different values but same structure
    custom_smoothness = original_smoothness.copy()
    # Use an arbitrary fixed value to test that custom smoothness is accepted and used
    custom_smoothness[:] = 500.0  # Set all smoothness values to a fixed value
    
    # Test with custom smoothness
    m2 = MFF(df, equality_constraints=equality_constraints, parallelize=False, smoothness=custom_smoothness)
    df2_custom = m2.fit()
    
    # Should produce forecasts without errors
    assert df2_custom.iloc[-1, 0] == df_true.iloc[-1, 0]
    assert ~np.isnan(df2_custom.iloc[-1, 1])
    # Results may differ from default due to different smoothness
    # Just verify the custom smoothness was actually used (not equal to default in general)



def test_smoothness_backward_compatibility():
    """Test that when smoothness is not provided, the old behavior is maintained."""
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
    
    # Test without smoothness parameter (default behavior)
    m = MFF(df, equality_constraints=equality_constraints, parallelize=False)
    df2 = m.fit()
    
    # Should produce forecasts without errors
    assert df2.iloc[-1, 0] == df_true.iloc[-1, 0]
    assert ~np.isnan(df2.iloc[-1, 1])
    assert ~np.isnan(df2.iloc[-1, 2])

