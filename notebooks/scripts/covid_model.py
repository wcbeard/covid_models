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
from covid_scrape import lgs, pl
import model_transformations as mtx

data_dir = Path('../data')
load_states = mem.cache(cvs.load_states)

# %%
cvs.pull_and_save_ga_county()
cvs.pull_and_save_ga_agg()
dfct = cvs.load_ga_county()
ga_agg = cvs.load_ga_agg()

# %%
dfs = (
    load_states(date="03-25")
    .sort_values(["state", "date"], ascending=True)
    .reset_index(drop=1)
)
print(f"Max date {dfs.date.max()}")
dfs = cvs.process_state(dfs)
print(f"Current actual max date: {cvs.load_states().date.max()}")


# %%
@mem.cache
def load2(date='03224'):
    return pd.read_csv('https://coronadatascraper.com/timeseries-tidy.csv')


# %%
dfs2 = load2(date='03224')
def _proc(df):
    df = (
        df
        .drop(['lat', 'long', 'url', ], axis=1)
        .query("country == 'USA'")
    )
    return df

dfs2 = _proc(dfs2)

# %%
tx = dfs2.assign(null_cty=lambda x: x.county.isnull()).query("date == '2020-03-24'").query("state == 'TX'")

tx.groupby(['type', 'null_cty']).value.sum().unstack().fillna(0).astype(int)


# %% [markdown]
# # Models

# %% [markdown]
# ## Mortality model

# %%
def write_model(mbase, dfsim, form, plot):
    """
    Save formula, plots, vegalite spec, predicted data to `mbase`
    model dir.
    """
    dfsim.to_parquet(mbase / 'preds.pq')
    with open(mbase / 'form.txt', 'w') as fp:
        fp.write(f"Formula: {form}")
        
    with A.data_transformers.enable('default'):
        A.Chart.save(plot, str(mbase / 'preds.png'))
        A.Chart.save(plot, str(mbase / 'preds.json'))


# %%
# import pystan

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

[min_date] = dfd.query("daysi == 0").date

def dayi2date(i):
    return min_date + pd.Timedelta(days=i)


# %%
dfsim1_ = mtx.mk_sim_df1(dfd).assign(date=lambda x: x.daysi.map(dayi2date))

# Write to R; generate simulations
dfd.to_feather(data_dir / 'mort_0324.fth')
dfsim1_.to_feather(data_dir / 'mort_0324_sim.fth')

# %% [markdown]
# ## Model v1

# %%
# Read end result
dfsim_out = pd.read_feather('covid/data/mort_0320_sim_out.fth')
dfsim1 = dfsim1_.merge(dfsim_out, on='row').pipe(log_preds)

# %% [markdown]
# #### Plot Results

# %%
color = "state"
x = "daysi"
y = "pred_mu"


pred = dfsim1
actual = dfd.assign(pred_mu=lambda x: x["death"])


def deco(ch):
    h = ch.mark_line().encode(
        x=A.X(x, title=x),
        y=A.Y(y, title=y, scale=lgs),
        color=color,
        tooltip=[color, x, y],
    )
    return h


h_act = deco(Chart(actual))
h_est = deco(Chart(pred))
h_est_err = h_est.mark_errorband().encode(
    y="pred_q10", y2="pred_q90", x="daysi", color=color,
)

ch_act = h_act + h_act.mark_point()
ch_est = h_est + h_est.mark_point() + h_est_err


form1 = "bf(ldeaths ~ daysi + (1 | state)) + gaussian()"
save_ch = (ch_act + ch_est).properties(title=form1)
# save_ch

# %%
# m1_path = Path('covid/models/mortal/v1')
# write_model(m1_path, dfsim1, "bf(ldeaths ~ daysi + (1 | state)) + gaussian()", save_ch)

# %% [markdown]
# ## Model v2

# %% [markdown]
# ### Coefs

# %%
color = 'dvers'
x = 'state'
y = 'Estimate'

m2_coefs = (
    pd.read_feather(data_dir / "mort_m2_coef.fth")
    .sort_values("Est.Error", ascending=True)
    .reset_index(drop=1)
)

pdf = m2_coefs

h = Chart(pdf).mark_point().encode(
    x=A.X(x, title=x, sort=None),
    y=A.Y(y, title='Exponent'),
    tooltip=[x, y]
)

herr = h.mark_errorbar().encode(y=A.Y('Q2.5', title='Exponent'), y2='Q97.5')
# (h + h.mark_point()).interactive()
herr + h

# %%

# %% [markdown]
# ### Load simulated data

# %%
# dfsim_out = pd.read_feather(data_dir / 'mort_v2_sim_out.fth')
dfsim_out = pd.read_feather(data_dir / 'mort_0324_sim_out.fth')
dfsim2 = dfsim1_.merge(dfsim_out, on='row').pipe(log_preds).assign(date=lambda x: x.daysi.map(dayi2date))


# %%
def plot_preds_act(dfsim, dfd, form, x = "daysi", log=True):
    color = "state"
    yargs = {}
    if log:
        yargs['scale'] = lgs
    y = "pred_mu"

    pred = dfsim
    actual = dfd.assign(pred_mu=lambda x: x["death"])

    def deco(ch):
        h = ch.mark_line().encode(
            x=A.X(x, title=x),
            y=A.Y(y, title=y, **yargs),
            color=color,
            tooltip=[color, x, y],
        )
        return h

    h_act = deco(Chart(actual))
    h_est = deco(Chart(pred))
    h_est_err = h_est.mark_errorband().encode(
        y="pred_q10", y2="pred_q90", x=x, color=color
    )

    ch_act = h_act + h_act.mark_point()
    ch_est = h_est + h_est.mark_point() + h_est_err

    save_ch = (ch_act + ch_est).properties(title=form)
    return save_ch


states_low_deaths = dfd.groupby('state').death.max().pipe(lambda x: x[x < 10]).index.tolist()

# bf(ldeaths ~ (1 | state) + daysi + (0 + daysi | state))
form2 = "ldeaths ~ (1 | state) + daysi + (0 + daysi | state)"

pdfsim = dfsim2.query("state not in @states_low_deaths")
pdfd = dfd.query("state not in @states_low_deaths")

m2pl_lin = plot_preds_act(dfsim=pdfsim, dfd=pdfd, form=form2, x='date', log=0).properties(width=900, height=600)
m2pl_log = plot_preds_act(dfsim=pdfsim, dfd=pdfd, form=form2, x='date', log=1).properties(width=900, height=600)

# with A.data_transformers.enable('default'):
#     A.Chart.save(m2pl_lin, '../fig/m2_pred_lin_0324.png')

# with A.data_transformers.enable('default'):
#     A.Chart.save(m2pl_log, '../fig/m2_pred_log_0324.png')

# %%

# %%
dfd.date.max()

# %%
dfd[:3]

# %%
min_preds = '2020-03-24'
_cs = 'date state positive negative tot ldeaths death daysi'.split()
pdf_latest = dfd.query("date > @min_preds")[_cs]

color = 'state'
x = 'date'
y = 'death'


h = Chart(pdf_latest).mark_line().encode(
    x=A.X(x, title=x),
    y=A.Y(y, title=y, scale=lgs),
    color=color,
    tooltip=[color, x, y]
)

(h + h.mark_point()).interactive()

# %%
m2pl_log.properties(width=500, height=400).interactive()

# %% [markdown]
# ## Log and Linear

# %%
m2pl_lin & m2pl_log

# %%
p2 = pl(dfd, x='date', y='death')
p2

# %% [markdown]
# # Deaths ~ delayed infection

# %% [markdown]
# ## Model: autoregressive model of prob by state

# %%
# Delayed df w/ state data
# 4 days w/ recorded deaths
ddfs = cvs.filter_mortality(dfs, days_previous=4, min_death_days=4)

# %%

# %%
ddelayed[:3]

# %%

# %%
# mod = ms.compile('../models/mortal2.stan')


# %%

# %%
ddfs.query("pos_delayed > 0")[:3]

# %%

# %%
ddelayed = sm.Mpv1.preprocess_data(ddfs)
data = sm.Mpv1.mk_data(ddelayed)
# data = mk_data(ddelayed)
res = mod.sampling(data=data)


# %%
def update_perc_est(ddelayed, res):
    par_est = ms.extract_arrs(res, pars=['p_delay'])['p_delay'].quantile([.05, .5, .95])
    ddelayed['pdel05'] = par_est.loc[.05].values
    ddelayed['pdel50'] = par_est.loc[.5].values
    ddelayed['pdel95'] = par_est.loc[.95].values
    return ddelayed

ddelayed = update_perc_est(ddelayed, res)

# %%
ddelayed[:3]

# %%
color = 'state'
x = 'date'
# y = 'pdel50'
y = 'perc_delayed'

pdf = ddelayed

h = Chart(pdf).mark_line().encode(
    x=A.X(x, title=x),
    y=A.Y(y, title=y),
    color=color,
    tooltip=[color, x, y]
)

h2 = h.encode(y=c.prev_perc_delayed, tooltip=[color, x, y, c.prev_perc_delayed])
h2 = (h2 + h2.mark_point())

h3 = h.encode(y=c.pdel50, tooltip=[color, x, y, c.pdel50])
h3 = (h3 + h3.mark_point())

r1 = h3 + h.mark_errorband().encode(y='pdel05', y2='pdel95') | h2
r2 = (h + h.mark_point())
r1 & r2

# %%

# %% [markdown]
# ## Autoregressive delayed infection rate

# %%
ddfs = cvs.filter_mortality(dfs, days_previous=4, min_death_days=4)

# %%

# %%
mod = ms.compile('../models/mortal_delay.stan')


# %%
def mk_data(df):
    data = dict(
        N=len(df),
        # S=df.state.nunique(),
        npos_delayed=df.pos_delayed,
        ntot_delayed=df.tot_delayed,
        perc_delayed=df.perc_delayed,
        perc_delayed_prev=df.prev_perc_delayed,
        
        n_deaths=df.death,
        p_death_baseline=df.prev_perc_delayed,
    )
    return data

_ints = ["death", "total", "positive", "pos_delayed", "tot_delayed"]
_fs = "perc_delayed prev_perc_delayed ".split()
ddelayed = (
    ddfs[["state", "date"] + _ints + _fs]
    .query("pos_delayed > 0")
    .query("prev_perc_delayed > 0")
    .assign(**{icol: lambda x, icol=icol: x[icol].astype(int) for icol in _ints})
    .reset_index(drop=1)
)
data = mk_data(ddelayed)


# %%
# data['perc_delayed']

# %%
def update_perc_est(ddelayed, res, parname='p_delay', pref='pdel'):
    fmt = "{:02d}".format
    qs = [.05, .5, .95]
    par_est = ms.extract_arrs(res, pars=[parname])[parname].quantile(qs)
    for q in qs:
        ddelayed[pref + fmt(int(q * 100))] = par_est.loc[q].values
    
#     ddelayed['pdel05'] = par_est.loc[.05].values
#     ddelayed['pdel50'] = par_est.loc[.5].values
#     ddelayed['pdel95'] = par_est.loc[.95].values
    return ddelayed


# %%
ddelayed = update_perc_est(ddelayed, res)
ddelayed = update_perc_est(ddelayed, res, parname='b_death', pref='bdeath')

# %%
ddelayed[:3]

# %%
res = mod.sampling(data=data)
par_p_delay = ms.extract_arrs(res, pars='p_delay')['p_delay']


# %%
def pipe(h, f):
    return f(h)

A.Chart.pipe = pipe

def add_point(h):
    return h + h.mark_point()

def add_line(h):
    return h + h.mark_line()


# %%
h.pipe(add_point)

# %%
color = 'state'
x = 'date'
# y = 'pdel50'
y = 'perc_delayed'

pdf = ddelayed

h = Chart(pdf).mark_line().encode(
    x=A.X(x, title=x),
    y=A.Y(y, title=y),
    color=color,
    tooltip=[color, x, y]
)

h2 = h.encode(y=c.prev_perc_delayed, tooltip=[color, x, y, c.prev_perc_delayed])
h2 = (h2 + h2.mark_point())

h3 = h.encode(y=c.pdel50, tooltip=[color, x, y, c.pdel50])
h3 = (h3 + h3.mark_point())

r1 = h3 + h.mark_errorband().encode(y='pdel05', y2='pdel95') | h2
r2 = (h + h.mark_point())

r3 = h.encode(y='bdeath50').pipe(add_point) + h.mark_errorband().encode(y='bdeath05', y2='bdeath95')
# bdeath05 	bdeath50 	bdeath95

r1 & r2
r3

# %%
b_delay = ms.extract_arrs(res)['b_death']
b_delay_qs = b_delay.quantile([.05, .5, .95])

# %%
plt.figure(figsize=(10, 10)); ax = plt.gca()
ms.plot_pctiles(b_delay)

# %%
par_p_delay

# %%
pl(ddelayed.query("prev_perc_delayed > 0"), y="perc_delayed", logy=0)


# %%
pdf = ddfs.query("positive > 0").query("n_pct_rows > 3")

# %%
pdf = ddfs.query("positive > 0").query("n_pct_rows > 3").query("tot_delayed > 0")
p1 = pl(pdf, y="death", x="positive", color="state", tt=['perc'])
p2 = pl(pdf.query("~dupe_neg"), y="death", x="pos_delayed")
p3 = pl(pdf.query("~dupe_neg"), y="pos_delayed", x="tot_delayed", logy=1, logx=1)
p1 | p2 | p3

# %%
pl(pdf, color="state") | pl(pdf.query("~dupe_neg"), y="negative") |  pl(pdf.query("~dupe_neg"), y="perc", logy=0)

# %%
ddfs[:3]

# %%
pdf = (
    ddfs.assign(perc_dth=lambda x: x.death / x.pos_delayed)
#     [["state", "perc_dth"]]
#     .drop_duplicates()
#     .sort_values("perc_dth", ascending=True)
)
pl(pdf.query("perc_dth > 0"), y="perc_dth", logy=0) | pl(pdf.query("perc_delayed > 0"), y="perc_delayed", logy=0)

# %%
pdf = dfs.query("positive > 0").query("n_pct_rows > 3")


# %%
pl(pdf, color="state") | pl(pdf.query("~dupe_neg"), y="negative") |  pl(pdf.query("~dupe_neg"), y="perc", logy=0)
