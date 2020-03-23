# ---
# jupyter:
#   jupytext:
#     formats: ipynb,scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%

# %%
from boot_utes import (reload, add_path, path, run_magics)
add_path('..', '../src/', '~/repos/myutils/', )

from src.crash_imps import *; exec(pu.DFCols_str); exec(pu.qexpr_str); run_magics()
# import utils.en_utils as eu; import data.load_data as ld; exec(eu.sort_dfs_str)

sns.set_style('whitegrid')


A.data_transformers.enable('json', prefix='../data/altair-data')
A.data_transformers.enable('default')

S = Series; D = DataFrame

from big_query import bq_read

import warnings
from matplotlib import MatplotlibDeprecationWarning

from joblib import Memory

mem = Memory(location="cache", verbose=0)

import altair.vegalite.v3 as A
import requests
pd.options.display.max_rows = 100
pd.options.display.min_rows = 40

# %% [markdown]
# # Load

# %%
import covid_scrape as cvs
from covid_scrape import pl, lgs
load_states = mem.cache(cvs.load_states)

# %%
cvs.pull_and_save_ga_county()
cvs.pull_and_save_ga_agg()

# %% [markdown]
# ## Download State data
#
# [Google doc](https://docs.google.com/spreadsheets/u/2/d/e/2PACX-1vRwAqp96T9sYYq2-i7Tj0pvTf6XVHjDSMIKBdZHXiCGGdNC0ypEU9NbngS8mxea55JuCFuua1MUeOj5/pubhtml#)

# %%
cvs.load_states().date.max()

# %%
dfs = load_states(date='03-20').sort_values(["state", 'date'], ascending=True).reset_index(drop=1)
print(f"Max date {dfs.date.max()}")
dfs = cvs.process_state(dfs)

# %%
dfct = cvs.load_ga_county()
dfct[:3]

# %%
ga_agg = cvs.load_ga_agg()
ga_agg[:3]

# %% [markdown]
# # Plot

# %% [markdown]
# ## By region

# %%
ctrl_sts = ('WA', 'CA')
south_sts = ('GA', 'FL', 'AL', 'NC', 'TN') + ctrl_sts
mw_sts = ('IN', 'MI', 'KY', 'OH', 'MO')
sth = dfs.query("state in @south_sts")
mw = dfs.query("state in @mw_sts")
region = dfs.query("state in @south_sts or state in @mw_sts")

# %% [markdown]
# ## Top States

# %%
pdf = dfs.query("state in ('WA', 'NY', 'CA')")
p1 = pl(pdf, color='state', x='date', y='positive', ii=True, logy=True)
p2 = pl(pdf.query("death > 0"), color='state', x='date', y='death', ii=True, logy=True)
p3 = pl(pdf.query("death > 0"), color='state', x='positive', y='death', ii=True, logy=True)
p1 | p2 | p3

# %% [markdown]
# ### Mortality

# %%
dfd = cvs.filter_mortality(dfs)

# %%
p1 = pl(dfd, color='state', x='pos_delayed', y='death', ii=True, logy=1)
p2 = pl(dfd, color='state', x='positive', y='death', ii=True, logy=1, logx=1, tt=['date'])
# p2 = pl(dfd, color='state', x='positive', y='death', ii=True, logy=True)
p1 | p2

# %%
p1 = pl(dfd, color='state', x='date', y='death', ii=True, logy=True)
p1

# %% [markdown]
# # Models

# %% [markdown]
# ## Mortality model

# %%
import pystan

def log_preds(df):
    preds = [c for c in df if c.startswith('pred_')]
    df = df.assign(**{pred: lambda x, pred=pred: 10 ** x[pred] for pred in preds})
    return df

dfd = cvs.filter_mortality(dfs)

dfd = (
    dfd.assign(
        ldeaths=lambda x: x.death.pipe(np.log10),
        daysi=lambda x: (x.date - x.date.min()).astype('timedelta64[D]').astype(int)
    )
    .reset_index(drop=1)
    .drop(["dateChecked", 'pending'], axis=1)
)

# dfd.to_feather('covid/data/mort_0320.fth')

# %%
def mk_sim_df1(dfd):
    base = dfd.groupby(["state"]).daysi.max().reset_index(drop=0)
    return pd.concat(
        [base.assign(daysi=lambda x: x.daysi.add(i)) for i in range(10)],
        axis=0,
        ignore_index=True,
    ).assign(row=lambda x: range(len(x)))


dfsim1_ = mk_sim_df1(dfd)
# dfsim1_.to_feather('covid/data/mort_0320_sim.fth')

# %%
dfsim1[:3]

# %% [markdown]
# ## Predictions

# %%
dfsim_out = pd.read_feather('covid/data/mort_0320_sim_out.fth')

# %%
dfsim1 = dfsim1_.merge(dfsim_out, on='row').pipe(log_preds)

# %%
dfs[:3]

# %%
pdf = dfs.query("positive > 0").query("n_pct_rows > 3")

# %%
pl(pdf, color="state") | pl(pdf.query("~dupe_neg"), y="negative") |  pl(pdf.query("~dupe_neg"), y="perc", logy=0)

# %% [markdown]
# ## Midwest

# %%
mdf = mw.query("positive > 0")

# %%
mdf[:3]

# %%
pl(mdf, color="state").properties() | pl(mdf.query("~dupe_neg"), y="negative") |  pl(mdf.query("~dupe_neg"), y="perc", logy=0)

# %%
pl(mdf, color="state").properties() | pl(mdf.query("death > 0"), y="death") |  pl(mdf.query("~dupe_neg"), y="perc", logy=0)

# %%
pl(sth.query("positive > 0"), color="state").properties() | pl(sth.query("death > 0"), y="death") |  pl(sth.query("~dupe_neg"), y="perc", logy=0)

# %%
pl(sth.query("positive > 0"), color="state").properties() | pl(sth.query("negative > 0"), y="negative") |  pl(sth.query("~dupe_neg"), y="perc", logy=0)

# %%
