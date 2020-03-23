import datetime as dt
from pathlib import Path
import re
import simplejson

import altair.vegalite.v3 as A
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np

pth = Path("~/repos/covid/data").expanduser()
url_ga_county = url_ga = "https://dph.georgia.gov/covid-19-daily-status-report"


# Utils
def nonull_tdd(s):
    """
    Gets timedelta in days for non-null elements. The
    rest are nan.
    """
    nonull_tdd.s = s
    nn_bm = s == s
    # print(s[nn_bm])
    # print(s[nn_bm].astype("timedelta64[D]"))
    tdd = s[nn_bm] / np.timedelta64(1, "D")
    res = pd.Series([np.nan] * len(s), index=s.index)
    res.loc[nn_bm] = tdd
    # print(res)
    return res


def try_drop(df, cs):
    cs = [cs] if isinstance(cs, str) else cs
    for c in cs:
        if c in df.columns:
            df = df.drop(c, axis=1)
    return df


#############
# GA county #
#############
def pull_ga_county(process=True):
    today = dt.date.today()
    tables = pd.read_html(url_ga_county)
    [table_] = [t for t in tables if "County" in t]
    if not process:
        return table_
    table = (
        try_drop(table_, ["Unnamed: 0"])
        .assign(date=today)
        .rename(columns=str.lower)
    )
    return table


def same_file(fout, df):
    dfo = pd.read_csv(fout)
    if len(dfo) != len(df):
        return False
    return sorted(df.cases) == sorted(dfo.cases)


def save_ga_county(df):
    date = df["date"].iloc[0]
    fout_yest = pth / f"{date - pd.Timedelta(days=1)}.csv"
    fout = pth / f"county-{date}.csv"

    yesterday_written = fout_yest.exists() and same_file(fout_yest, df)
    if yesterday_written:
        print("written yest")
        return fout_yest

    today_written = fout.exists() and same_file(fout, df)
    if today_written:
        print("written today")
        return fout

    print("New data!")
    df.to_csv(fout, index=False)

    return fout


def pull_and_save_ga_county():
    df = pull_ga_county()
    fout = save_ga_county(df)
    return fout


def load_ga_county():
    fns = pth.glob("county-*.csv")

    def fn_date(x):
        ln = len("county-")
        return x[ln:]

    res = (
        pd.concat(
            [pd.read_csv(fn).assign(fn_date=fn_date(fn.stem)) for fn in fns],
            ignore_index=True,
        )
        .drop_duplicates()
        .rename(columns={"date": "pull_date"})
        .assign(date=lambda x: x[["pull_date", "fn_date"]].min(axis=1))
        .sort_values(["date", "cases"], ascending=[True, False])
        .reset_index(drop=1)
    )
    return res


######################
# Georgia aggregates #
######################
def get_generated_date(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, features="lxml")
    [rep_gen] = [
        phrase
        for em in soup.select("em")
        for phrase in em.contents
        if "Report generated" in phrase
    ]
    [date_str] = re.findall(r"\d+/\d+/\d+", rep_gen)
    return date_str


def get_ga_agg():
    """
    Table on page expected to have columns
    'Lab' 'Number of Positive Tests' 'Total Tests'
    """
    tables = pd.read_html(url_ga)
    [table] = [t for t in tables if "Lab" in t]

    gen_date = get_generated_date(url_ga)

    table = table.assign(date_gen=gen_date).assign(
        date_gen=lambda x: pd.to_datetime(x.date_gen)
    )
    return table


def save_ga_agg(df):
    date = df.date_gen.dt.date.iloc[0]
    fn = pth / f"ga-{str(date)}.csv"
    if fn.exists():
        print(f"File for {date} already exists!")
        return fn
    df.to_csv(fn)
    return fn


def load_ga_agg():
    fns = pth.glob("ga-*.csv")
    res = (
        pd.concat([pd.read_csv(fn) for fn in fns], ignore_index=True)
        .drop(["Unnamed: 0"], axis=1)
        .rename(
            columns={"Number of Positive Tests": "pos", "Total Tests": "tot"}
        )
        .rename(columns=str.lower)
    )
    return res


def pull_and_save_ga_agg():
    df = get_ga_agg()
    fn = save_ga_agg(df)
    return fn


###################
# Load State data #
###################
def load_states(date="03-16"):
    """
    DataFrameate doesn't mean anyting. Just for caching
    """
    r = requests.get("http://covidtracking.com/api/states/daily")
    return pd.DataFrame(simplejson.loads(r.content))


###########
# Process #
###########
def str2float(s):
    s = str(s)
    return float(s.replace(",", ""))


def duped_neg(neg):
    return neg == neg.shift(1)


def process_state(df):
    res = (
        df.assign(
            date=lambda x: pd.to_datetime(x.date.astype(str)),
            negative=lambda x: x.negative.map(str2float),
        )
        .assign(
            perc=lambda x: x.positive / (x.positive + x.negative),
            dupe_neg=lambda df: df.groupby(["state"]).negative.transform(
                duped_neg
            ),
            # di_tot=lambda x: x.groupby(["state"]).apply(align_total)
            #     align_time(x, 'total', 26),
        )
        .assign(n_pct_rows=lambda df: percent_col(df))
    )
    return res


def useful_percent_row(df):
    return df.eval("perc > 0 & positive > 10")


def percent_col(df):
    return (
        df.assign(num_pct_rows=lambda x: useful_percent_row(x))
        .groupby(["state"])
        .num_pct_rows.transform("sum")
        .astype(int)
    )


# Mortality
def mkshft(n):
    def shift(s):
        return s.shift(n)

    return shift


def filter_mortality(dfs, days_previous=4, min_death_days=4):
    """
    Get list of states with at least `min_death_days`
    rows of mortality data.
    Add column `pos_delayed` of positive cases `days_previous`
    days ago.
    """
    mortal_states = (
        dfs.query("death > 0")
        .state.value_counts(normalize=0)
        .pipe(lambda x: x[x > min_death_days])
        .index.tolist()
    )
    dfd = (
        dfs.query(f"state in {mortal_states}")
        .query("death > 3")
        .assign(
            pos_delayed=lambda df: df.groupby(["state"]).positive.transform(
                mkshft(days_previous)
            )
        )
    )
    return dfd


# Plot
lgs = A.Scale(type="log", zero=False)


def pl(
    pdf,
    color="state",
    x="date",
    y="positive",
    ii=True,
    logy=True,
    logx=False,
    tt=[],
):
    tt = list(tt)
    ykw = dict(scale=lgs) if logy else {}
    xkw = dict(scale=lgs) if logx else {}
    h = (
        A.Chart(pdf)
        .mark_line()
        .encode(
            x=A.X(x, title=x, **xkw),
            y=A.Y(y, title=y, **ykw),
            color=color,
            tooltip=[color, x, y] + tt,
        )
    )
    hh = h + h.mark_point()
    if ii:
        return hh.interactive()
    return hh
