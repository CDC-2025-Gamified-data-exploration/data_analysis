# This file contains the function for plotting a histogram.

import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram(series, bins='auto'):
    """
    Plots a histogram for a continuous variable.

    Args:
        series (pandas.Series): The data series to plot.
        bins (int or str): The number of bins or a binning strategy string.
    """
    sns.histplot(series, bins=bins)
    plt.title(f'Histogram of {series.name}')
    plt.xlabel(series.name)
    plt.ylabel('Frequency')
    plt.show()
