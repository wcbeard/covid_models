import pandas as pd


def mk_sim_df1(dfd):
    base = dfd.groupby(["state"]).daysi.max().reset_index(drop=0)
    return pd.concat(
        [base.assign(daysi=lambda x: x.daysi.add(i)) for i in range(10)],
        axis=0,
        ignore_index=True,
    ).assign(row=lambda x: range(len(x)))
