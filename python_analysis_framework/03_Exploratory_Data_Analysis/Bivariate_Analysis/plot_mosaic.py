# This file contains the function for plotting a mosaic plot.

from statsmodels.graphics.mosaicplot import mosaic
import matplotlib.pyplot as plt

def plot_mosaic(df, index_cols):
    """
    Creates a mosaic plot to visualize associations between two or more categorical variables.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        index_cols (list of str): The names of the categorical columns to plot.
    """
    mosaic(df, index=index_cols)
    plt.show()
