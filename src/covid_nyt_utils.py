import numpy as np
import pandas as pd


def consecutive_dates(ds):
    """Return True if all dates in `ds` are ordered
    and consecutive.
    """
    diff = ds - ds.shift(1)
    return diff.astype("timedelta64[D]").fillna(1).eq(1).all()


def diff(cs):
    return cs - cs.shift(1)


def days_since_first(ds):
    return (ds - ds.min()).astype("timedelta64[D]").astype(int)


def proc(df):
    df = (
        df.assign(date=lambda x: pd.to_datetime(x.date))
        .sort_values(["state", "date"], ascending=True)
        .reset_index(drop=1)
        .assign(
            cases_new=lambda df: df.groupby(["state"]).cases.transform(diff),
            deaths_new=lambda df: df.groupby(["state"]).deaths.transform(diff),
        )
        .assign(
            ldeaths=lambda x: x["deaths"].pipe(np.log),
            lcases=lambda x: x["cases"].pipe(np.log),
            ldeaths_new=lambda x: x["deaths_new"].pipe(np.log),
            lcases_new=lambda x: x["cases_new"].pipe(np.log),
        )
    )

    df["min_date_d10"] = txf_min_date(df.query("deaths >= 10"), df)
    df["days_since_d10"] = (
        (df.date - df.min_date_d10)
        .astype("timedelta64[D]")
        .fillna(-1)
        .astype(int)
    )
    df = df.drop(["min_date_d10"], axis=1)
    return df


def txf_min_date(df_filt, df_all, by="state"):
    state2min_date = (
        df_filt.query("deaths > 10").groupby([by]).date.min().to_dict()
    )
    return df_all[by].map(state2min_date)
