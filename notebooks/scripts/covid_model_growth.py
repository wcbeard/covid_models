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
from boot_utes import (reload, add_path, path, run_magics)
add_path('..', '../src/', '~/repos/myutils/', )

from src.crash_imps import *; exec(pu.DFCols_str); exec(pu.qexpr_str); run_magics()
# import utils.en_utils as eu; import data.load_data as ld; exec(eu.sort_dfs_str)

sns.set_style('whitegrid')


# A.data_transformers.enable('json', prefix='../data/altair-data')
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

# %%
import pystan
import models.stan_mods as sm

# %% [markdown]
# # Load

# %%
import covid_scrape as cvs
from covid_scrape import add_point, add_line, A
from covid_scrape import lgs, pl
import model_transformations as mtx

data_dir = Path('../data')
# load_states = mem.cache(cvs.load_states)

dfs_ = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv')

# %%
# date_arg = '03-29'
# fn_date = '0329'
import covid_nyt_utils as cnu

dfs = cnu.proc(dfs_)

non_consecutive_states = dfs.groupby(['state']).date.agg(cnu.consecutive_dates).pipe(lambda x: x[~x]).tolist()
print(f"non_consecutive_states: {non_consecutive_states}")

print(f"Max date {dfs.date.max()}")
# print(f"Current actual max date: {cvs.load_states().date.max()}")

# %% [markdown]
# # Models
#
# ## Growth

# %%

# %%
dfs[:3]

# %%

# %%
color = "state"
x = "date"
y = "deaths"


pdf = dfs.query("days_since_d10 >= 0 & deaths >= 10 & deaths_new > 0")

h_date_ = (
    Chart(pdf)
    .mark_line()
    .encode(
        x=A.X(x, title=x),
        y=A.Y(y, title=y, scale=A.Scale(type="log", zero=False)),
        color=color,
        tooltip=[color, x, y],
    )
)

h_zeroed = h_date_.encode(x=c.days_since_d10).pipe(add_point)
h_dd_ = h_date_.encode(
    x=A.X("cases", scale=A.Scale(type="log", zero=False)),
    y=A.Y("cases_new", scale=A.Scale(type="log", zero=False)),
    tooltip=[color, x, 'deaths', 'deaths_new'],
)

h_dd = h_dd_.pipe(add_point)
# .pipe(add_point)

h_date = h_date_.pipe(add_point)
(h_date).interactive() | h_zeroed.interactive() | h_dd.interactive()

# %%
color = 'state'
# x = "date"

# y = ''



h_dd = Chart(pdf).mark_line().encode(
    x=A.X("deaths", scale=A.Scale(type="log", zero=False)),
    y=A.Y("deaths_new", scale=A.Scale(type='log')),
    tooltip=[color, 'deaths', 'deaths_new'],
)


h_dd.pipe(add_point)

# %%
h_dd.to_dict()

# %%
dfs.groupby(['state']).deaths.max().sort_values(ascending=False)

# %%
dfs.groupby(['state']).date.transform(cnu.days_since_first)

# %%

# %%
for state, gdf in dfs.groupby(['state']):
    break

# %%

# %%
gdf.cases.reset_index(drop=0).assign(nc=lambda x: new_cases(x.cases))

# %%
gdf.date.shift(1)

# %%
dfs[:3]
