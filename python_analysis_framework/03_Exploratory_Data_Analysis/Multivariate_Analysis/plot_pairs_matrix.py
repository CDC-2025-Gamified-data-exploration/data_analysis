# This file contains the function for plotting a pairs matrix (scatterplot matrix).

import seaborn as sns
import matplotlib.pyplot as plt

def plot_pairs_matrix(df, hue_col=None):
    """
    Plots a matrix of scatterplots for pairwise relationships in a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to plot.
        hue_col (str, optional): A column name to use for coloring the points.
    """
    sns.pairplot(df, hue=hue_col)
    plt.show()
