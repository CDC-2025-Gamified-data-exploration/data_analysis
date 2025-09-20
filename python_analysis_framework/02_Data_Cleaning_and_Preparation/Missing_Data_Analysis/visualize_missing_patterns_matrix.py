# This file contains the function for a matrix visualization of missing data.

import missingno as msno
import matplotlib.pyplot as plt

def visualize_missing_patterns_matrix(df):
    """
    Creates a matrix plot to visualize the location of missing data.
    This is conceptually similar to VIM::matrixplot in R.

    Args:
        df (pandas.DataFrame): The DataFrame to analyze.
    """
    msno.matrix(df)
    plt.show()
