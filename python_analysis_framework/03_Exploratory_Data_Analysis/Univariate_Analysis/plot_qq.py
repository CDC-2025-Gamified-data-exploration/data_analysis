# This file contains the function for creating a Q-Q plot.

import statsmodels.api as sm
import matplotlib.pyplot as plt

def plot_qq(series, dist='norm'):
    """
    Generates a Q-Q plot to check if a sample follows a specific distribution.

    Args:
        series (pandas.Series): The data series to plot.
        dist (str or statsmodels.api.distributions): The distribution to test against.
    """
    sm.qqplot(series, dist=dist, line='s')
    plt.title(f'Q-Q Plot for {series.name}')
    plt.show()
