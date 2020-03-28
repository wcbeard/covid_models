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


A.data_transformers.enable('json', prefix='../data/altair-data')
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
# from covid_scrape import pl, lgs
load_states = mem.cache(cvs.load_states)

# %%
cvs.pull_and_save_ga_county()
cvs.pull_and_save_ga_agg()

# %% [markdown]
# ## Download State data
#
# [Google doc](https://docs.google.com/spreadsheets/u/2/d/e/2PACX-1vRwAqp96T9sYYq2-i7Tj0pvTf6XVHjDSMIKBdZHXiCGGdNC0ypEU9NbngS8mxea55JuCFuua1MUeOj5/pubhtml#)

# %%
# Check that this changes
cvs.load_states().date.max()

# %%
# Then increment arg
date_arg = '03-27'
dfs = load_states(date=date_arg).sort_values(["state", 'date'], ascending=True).reset_index(drop=1)
dfs.date.max()

# %%
dfct = cvs.load_ga_county()
dfct[:3]

# %%
ga_agg = cvs.load_ga_agg()
ga_agg[:3]

# %%
