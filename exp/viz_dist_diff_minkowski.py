"""
Experiment summary
------------------
Treat each province/state in a country cases over time
as a vector, do a simple K-Nearest Neighbor between
countries. What country has the most similar trajectory
to a given country?
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

    other_region = dist['minkowski'][0]
    regions = [region, other_region]
    for val in regions:
        df = data.filter_by_attribute(
            confirmed, "Country/Region", val)
        cases, labels = data.get_cases_chronologically(df)
        cases = cases.sum(axis=0)

        lines = ax.plot(cases, label=labels[0,1])

    ax.set_ylabel('# of confirmed cases')
    ax.set_xlabel("Time (days since Jan 22, 2020)")
    ax.set_yscale('log')
    ax.legend()
    plt.tight_layout()
    region = region.replace('*', '')
    other_region = other_region.replace('*', '')
    plt.title(f'Comparing confirmed cases in {region} and {other_region}')
    plt.savefig(f'results/dist_diff_minkowski/{region}.png')
    plt.close()
    print(region)