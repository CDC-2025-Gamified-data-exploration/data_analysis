# This file contains the function for plotting a box-and-whisker plot.

import matplotlib.pyplot as plt
import seaborn as sns

def plot_boxplot(series):
    """
    Plots a box-and-whisker plot for a continuous variable.

    Args:
        series (pandas.Series): The data series to plot.
    """
    sns.boxplot(y=series)
    plt.title(f'Box Plot of {series.name}')
    plt.ylabel(series.name)
    plt.show()
