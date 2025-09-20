# This file contains the function for plotting a pie chart.

import matplotlib.pyplot as plt

def plot_piechart(series):
    """
    Plots a pie chart for a categorical variable.

    Args:
        series (pandas.Series): The categorical data series.
    """
    counts = series.value_counts()
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
    plt.title(f'Pie Chart of {series.name}')
    plt.ylabel('') # Hides the y-label which is often redundant
    plt.show()
