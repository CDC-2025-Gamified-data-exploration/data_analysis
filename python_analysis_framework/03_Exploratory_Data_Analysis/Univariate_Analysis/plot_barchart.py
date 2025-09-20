# This file contains the function for plotting a bar chart.

import matplotlib.pyplot as plt
import seaborn as sns

def plot_barchart(series):
    """
    Plots a bar chart for a categorical variable.

    Args:
        series (pandas.Series): The categorical data series.
    """
    sns.countplot(y=series)
    plt.title(f'Bar Chart of {series.name}')
    plt.xlabel('Count')
    plt.ylabel(series.name)
    plt.show()
