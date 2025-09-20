# This file contains the function for plotting a hexagonal binning plot.

import matplotlib.pyplot as plt

def plot_hexbin(df, x_col, y_col, gridsize=50):
    """
    Plots a hexagonal binning plot for two continuous variables, useful for large datasets.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        x_col (str): The name of the column for the x-axis.
        y_col (str): The name of the column for the y-axis.
        gridsize (int): The number of hexagons in the x-direction.
    """
    df.plot.hexbin(x=x_col, y=y_col, gridsize=gridsize)
    plt.title(f'Hexbin Plot of {y_col} vs. {x_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()
