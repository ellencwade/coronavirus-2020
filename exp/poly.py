import sys
sys.path.insert(0, '..')

from utils import data
import os
import sklearn
import numpy as np
import json
import matplotlib.pyplot as plt

# ------------ HYPERPARAMETERS -------------
BASE_PATH = '../COVID-19/csse_covid_19_data/'
# degrees variable sets number of degrees of the polynomial
degrees = 4
# ------------------------------------------

confirmed = os.path.join(
    BASE_PATH,
    'csse_covid_19_time_series',
    'time_series_covid19_confirmed_global.csv')
confirmed = data.load_csv_data(confirmed)

for val in np.unique(confirmed["Country/Region"]):
    plt.style.use('fivethirtyeight')

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    cm = plt.get_cmap('jet')

    df = data.filter_by_attribute(
        confirmed, "Country/Region", val)
    cases, labels = data.get_cases_chronologically(df)
    cases = cases.sum(axis=0)

    lines = ax.plot(cases, label=val)
    lines[0].set_linestyle('solid')
    lines[0].set_color('blue')

    new_cases = cases
    total_days = len(cases)
    start_day = len(new_cases[new_cases == 0])
    new_cases = new_cases[new_cases > 0]

    x = np.linspace(start_day, len(cases)-1, total_days-start_day)
    poly = np.polyfit(x, new_cases.astype(float), degrees)
    p = np.poly1d(poly)
    x2 = np.linspace(start_day, total_days + 30, total_days+30-start_day+1)
    predictions = p(x2)
    zeros = np.zeros(start_day)
    predictions = np.concatenate([zeros, predictions])
    print(x2[x2.size - 1])
    print(predictions[predictions.size - 1])

    lines = ax.plot(predictions, label='Predicted')
    lines[0].set_linestyle('dotted')
    lines[0].set_color('black')

    ax.set_ylabel('# of confirmed cases')
    ax.set_xlabel("Time (days since Jan 22, 2020)")

    ax.set_yscale('log')
    ax.legend()
    plt.tight_layout()
    plt.title('Predicted total # of cases')
    val = val.replace('*', '')
    plt.savefig('results/polynomial_predictions_{0}/{1}.png'.format(degrees, val))
    plt.close()

    print(val)