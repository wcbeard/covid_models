from collections import Counter, defaultdict, OrderedDict
import datetime as dt
from importlib import reload
import itertools as it
from itertools import count
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import numpy.random as nr
import scipy as sp
import pandas as pd
from pandas import DataFrame, Series
from pandas.testing import assert_frame_equal, assert_series_equal
import toolz.curried as z
from numba import njit
import fastparquet

from functools import wraps, partial, reduce, lru_cache
import operator as op
from operator import itemgetter as sel, attrgetter as prop

from glob import glob
import re, sys, os, time
from os.path import *
from os import path

# !pip install simplejson
import simplejson
from pathlib import Path

from pyarrow.parquet import ParquetFile as Paf
from fastparquet import ParquetFile

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ss = lambda x: StandardScaler().fit_transform(x.values[:, None]).ravel()

sns.set_palette("colorblind")


# Myutils
import myutils as mu

reload(mu)
import mystan as ms
import pandas_utils as pu
import pandas_utils3 as p3
from faster_pandas import MyPandas as fp

ap = mu.dmap

import dask.dataframe as dd
import dask

dd.DataFrame.q = lambda self, q, local_dict={}, **kw: self.query(
    q, local_dict=z.merge(local_dict, kw)
)

import altair as A
from altair import Chart, expr as E, datum as D

vc = z.compose(Series, Counter)

from plotnine import ggplot, qplot, aes, theme, ggtitle, xlab, ylab
import plotnine as p9


def mk_geom(p9, pref="geom_"):
    geoms = [c for c in dir(p9) if c.startswith(pref)]
    geom = lambda: None
    geom.__dict__.update(
        {name[len(pref) :]: getattr(p9, name) for name in geoms}
    )
    return geom


geom = mk_geom(p9, pref="geom_")
facet = mk_geom(p9, pref="facet_")


def run_magics():
    args = [
        "matplotlib inline",
        "autocall 1",
        "load_ext autoreload",
        "autoreload 2",
    ]
    for arg in args:
        get_ipython().magic(arg)


DataFrame.sel = lambda df, f: df[[c for c in df if f(c)]]
Path.g = lambda self, *x: list(self.glob(*x))

pd.options.display.width = 220


lrange = z.compose(list, range)
lmap = z.compose(list, map)
lfilter = z.compose(list, filter)


def read(fn, mode="r"):
    with open(fn, mode) as fp:
        txt = fp.read()
    return txt
