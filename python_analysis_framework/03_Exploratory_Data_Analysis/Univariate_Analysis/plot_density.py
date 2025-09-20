# This file contains the function for plotting a kernel density estimate.

import matplotlib.pyplot as plt
import seaborn as sns

def plot_density(series):
    """
    Plots a Kernel Density Estimate (KDE) for a continuous variable.

    Args:
        series (pandas.Series): The data series to plot.
    """
    sns.kdeplot(series)
    plt.title(f'Density Plot of {series.name}')
    plt.xlabel(series.name)
    plt.ylabel('Density')
    plt.show()
