"""
Experiment summary
------------------
Plots 1 FIPS' data from time_series_covid19_confirmed_US data
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
    'time_series_covid19_confirmed_US.csv')
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

plt.style.use('fivethirtyeight')

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)
cm = plt.get_cmap('jet')

FIPS = 36061
df = data.filter_by_attribute(
    confirmed, "FIPS", FIPS)
cases, labels = data.get_cases_chronologically(df)
cases = cases[0, 7:]

FIPS = '36061'
lines = ax.plot(cases)

ax.set_ylabel('# of confirmed cases')
ax.set_xlabel("Time (days since Jan 22, 2020)")
ax.set_yscale('log')
plt.tight_layout()
plt.title(f'Confirmed cases in {FIPS}')
plt.savefig(f'results/viz_FIPS/{FIPS}.png')
plt.close()
print(FIPS)