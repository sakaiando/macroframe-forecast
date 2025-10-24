# Disclaimer: Reuse of this tool and IMF information does not imply
# any endorsement  of the research and/or product. Any research presented
# should not be reported as representing the views of the IMF,
# its Executive Board, member governments.

# Mix-frequency is not working properly yet, waiting for Pandas to fix error: https://github.com/pandas-dev/pandas/issues/59775

import pandas as pd

from .utils import (
    BreakDataFrameIntoTimeSeriesList,
    ConcatMixFreqMultiIndexSeries,
    DefaultForecaster,
    FillAllEmptyCells,
    GenLamstar,
    GenPredTrueData,
    GenSmoothingMatrix,
    GenVecForecastWithIslands,
    GenWeightMatrix,
    OrganizeCells,
    Reconciliation,
    StringToMatrixConstraints,
    get_freq_of_freq,
)

class MFF_mixed_freqency:
    def __init__(
        self, df_dict, forecaster=DefaultForecaster(), constraints_with_wildcard=[], ineq_constraints_with_wildcard=[]
    ):
        self.df_dict = df_dict
        self.forecaster = forecaster
        self.constraints_with_wildcard = constraints_with_wildcard
        self.ineq_constraints_with_wildcard = ineq_constraints_with_wildcard

    def fit(self):
        df_dict = self.df_dict
        forecaster = self.forecaster
        constraints_with_wildcard = self.constraints_with_wildcard
        # TODO: delete, the assignment below, if not needed
        ineq_constraints_with_wildcard = self.ineq_constraints_with_wildcard  # noqa: F841

        # create constraints
        freq_order = ["Y", "Q", "M", "W", "D", "H", "T", "S"]
        lowest_freq = freq_order[min([freq_order.index(k) for k in df_dict.keys()])]

        df0_list = []
        all_cells_list = []
        unknown_cells_list = []
        known_cells_list = []
        islands_list = []
        for k in df_dict.keys():
            df0_k, all_cells_k, unknown_cells_k, known_cells_k, islands_k = OrganizeCells(df_dict[k])
            df0_list.append(df0_k)
            all_cells_list.append(all_cells_k)
            unknown_cells_list.append(unknown_cells_k)
            known_cells_list.append(known_cells_k)
            islands_list.append(islands_k)

        df0_stacked = ConcatMixFreqMultiIndexSeries([df0.T.stack() for df0 in df0_list], axis=0)
        all_cells = pd.concat(all_cells_list, axis=0)
        unknown_cells = pd.concat(unknown_cells_list, axis=0)
        known_cells = pd.concat(known_cells_list, axis=0)
        islands = pd.concat(islands_list, axis=0)

        C, d = StringToMatrixConstraints(df0_stacked, all_cells, unknown_cells, known_cells, constraints_with_wildcard)

        # combine all frequncies into the lowest frequency dataframe
        df0wide_list = []
        df0wide_colflat_list = []
        for df in df0_list:
            df0 = df.copy()  # don't want to change df0_list
            df0_freq = df0.index.freqstr[0]

            if df0_freq == lowest_freq:
                df0wide_freq = df0.copy()
                df0wide_colfat_freq = pd.Series(df0wide_freq.columns, index=df0wide_freq.columns)

            else:
                index_freq = df0.index.asfreq(lowest_freq)
                col_freq = df0_freq + get_freq_of_freq(df0.index, df0_freq).astype(str)
                df0.index = pd.MultiIndex.from_arrays([index_freq, col_freq])
                df0wide_freq = df0.unstack()
                df0wide_colfat_freq = pd.Series(df0wide_freq.columns.map("_".join), index=df0wide_freq.columns)

            df0wide_list.append(df0wide_freq)
            df0wide_colflat_list.append(df0wide_colfat_freq)

        df0wide = pd.concat(df0wide_list, axis=1)
        # TODO: delete, the assignment below, if not needed
        df0wide_col = df0wide.columns  # noqa: F841
        df0wide_colflat = pd.concat(df0wide_colflat_list)

        # 1st step forecast
        df0wide.columns = df0wide_colflat.values.tolist()  # colname has to be single index
        df1wide, df1wide_model = FillAllEmptyCells(df0wide, forecaster)
        predwide, truewide, modelwide = GenPredTrueData(df0wide, forecaster)

        # get df1_list by breaking wide dataframe into different frequencies
        df1_list = []
        for df0i, df0 in enumerate(df0_list):
            if df0.index.freqstr[0] == lowest_freq:
                df1_freq = df0.copy()
                df1_freq.update(df1wide.loc[:, df0wide_colflat_list[df0i].values])
            else:
                df1wide_freq = df1wide.loc[:, df0wide_colflat_list[df0i].values]
                df1wide_freq.columns = pd.MultiIndex.from_tuples(df0wide_colflat_list[df0i].index)
                df1_freq = df0wide_list[df0i].copy().stack(future_stack=True)  # storage
                df1_freq.update(df1wide_freq.stack(future_stack=True))
                df1_freq.index = df0_list[df0i].index

            df1_list.append(df1_freq)

        # get pred_list, true_list by breaking dataframes into different frequencies
        pred_allfreq = []
        true_allfreq = []
        for df0i, df0 in enumerate(df0_list):
            # get nan cells
            df0wide_freq = df0wide_list[df0i].copy()
            df0wide_freq.columns = df0wide_colflat_list[df0i].values
            na_cells = df0wide_freq.isna()[df0wide_freq.isna()].T.stack().index

            # slice predwide
            pred_freq = predwide.loc[:, na_cells]
            true_freq = truewide.loc[:, na_cells]

            if df0.index.freqstr[0] != lowest_freq:
                # reshape colname multiindex of (var_freq,lowestfreq) to var_lowestfreqfreq
                colflat = pred_freq.columns
                var_list = [v[: v.rfind("_")] for v in colflat.get_level_values(0)]
                freq_list = [v[v.rfind("_") + 1 :] for v in colflat.get_level_values(0)]
                lowest_freq_list = colflat.get_level_values(-1).astype(str)
                original_time = pd.PeriodIndex(
                    [lowest_freq_list[i] + freq_list[i] for i in range(len(colflat))], freq=df0.index.freq
                )
                pred_freq_colname = pd.MultiIndex.from_arrays([var_list, original_time])
                pred_freq.columns = pred_freq_colname
                true_freq.columns = pred_freq_colname

            # change col order
            pred_freq = pred_freq.loc[:, df0.isna()[df0.isna()].T.stack().index]
            true_freq = true_freq.loc[:, pred_freq.columns]

            # append pred, true for each frequency
            pred_allfreq.append(pred_freq)
            true_allfreq.append(true_freq)

        # break dataframes in to lists
        ts_list = []
        pred_list = []
        true_list = []
        for df0i, df0 in enumerate(df0_list):
            ts_list_freq, pred_list_freq, true_list_freq = BreakDataFrameIntoTimeSeriesList(
                df0, df1_list[df0i], pred_allfreq[df0i], true_allfreq[df0i]
            )

            ts_list += ts_list_freq
            pred_list += pred_list_freq
            true_list += true_list_freq

        # get parts for reconciliation
        # islands_list_all_freq = pd.concat(islands_list)

        y1 = GenVecForecastWithIslands(ts_list, islands)
        W, shrinkage = GenWeightMatrix(pred_list, true_list)
        smoothness = GenLamstar(pred_list, true_list)
        Phi = GenSmoothingMatrix(W, smoothness)

        y2 = Reconciliation(y1, W, Phi, C, d)

        # reshape vector y2 into df2
        y2 = y2.T.stack(future_stack=True)
        y2.index = y2.index.droplevel(level=0)
        df2_list = []
        for df0 in df0_list:
            df2_freq = df0.copy()
            df2_freq.update(y2, overwrite=False)  # fill only nan cells of df0
            df2_list.append(df2_freq)

        self.df0_list = df0_list
        self.df1_list = df1_list
        self.df2_list = df2_list
        return self.df2_list
