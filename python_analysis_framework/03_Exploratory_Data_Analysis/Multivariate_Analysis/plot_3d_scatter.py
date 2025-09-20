# This file contains the function for plotting a 3D scatter plot.

import plotly.express as px

def plot_3d_scatter(df, x_col, y_col, z_col, color_col=None):
    """
    Creates an interactive 3D scatter plot.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        x_col (str): The column for the x-axis.
        y_col (str): The column for the y-axis.
        z_col (str): The column for the z-axis.
        color_col (str, optional): The column to use for coloring the points.
    """
    fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=color_col)
    fig.show()
