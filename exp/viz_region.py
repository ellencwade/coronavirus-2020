"""
Experiment summary
------------------
Plots 1 region's data
"""

import sys
sys.path.insert(0, '..')

from utils import data
import os
import sklearn
import numpy as np
import json
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# ------------ HYPERPARAMETERS -------------
BASE_PATH = '../COVID-19/csse_covid_19_data/'
# ------------------------------------------

confirmed = os.path.join(
    BASE_PATH,
    'csse_covid_19_time_series',
    'time_series_covid19_confirmed_global.csv')
confirmed = data.load_csv_data(confirmed)
features = []
targets = []

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)
cm = plt.get_cmap('jet')
NUM_COLORS = 0
LINE_STYLES = ['solid', 'dashed', 'dotted']
NUM_STYLES = len(LINE_STYLES)

dist_diff = os.path.join('../exp/results/', 'knn_dist_diff.json')
f = open(dist_diff,)
dist_diff = json.load(f)

for region, dist in dist_diff.items():
    plt.style.use('fivethirtyeight')

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    cm = plt.get_cmap('jet')

    val = region
    df = data.filter_by_attribute(
        confirmed, "Country/Region", val)
    cases, labels = data.get_cases_chronologically(df)
    cases = cases.sum(axis=0)

    lines = ax.plot(cases, label=val)

    ax.set_ylabel('# of confirmed cases')
    ax.set_xlabel("Time (days since Jan 22, 2020)")
    ax.set_yscale('log')
    ax.legend()
    plt.tight_layout()
    region = region.replace('*', '')
    plt.title(f'Confirmed cases in {region}')
    plt.savefig(f'results/viz_region/{region}.png')
    plt.close()
    print(region)