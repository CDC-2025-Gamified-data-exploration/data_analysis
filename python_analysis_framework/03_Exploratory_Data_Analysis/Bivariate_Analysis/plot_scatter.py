# This file contains the function for plotting a scatter plot.

import matplotlib.pyplot as plt
import seaborn as sns

def plot_scatter(df, x_col, y_col):
    """
    Plots a scatter plot for two continuous variables.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        x_col (str): The name of the column for the x-axis.
        y_col (str): The name of the column for the y-axis.
    """
    sns.scatterplot(data=df, x=x_col, y=y_col)
    plt.title(f'Scatter Plot of {y_col} vs. {x_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()
