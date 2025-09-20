# This file contains the function for an aggregate visualization of missing data.

import missingno as msno
import matplotlib.pyplot as plt

def visualize_missing_patterns_aggr(df):
    """
    Creates a bar chart visualizing the count of non-missing values in each column.
    This is conceptually similar to one of the plots from VIM::aggr in R.

    Args:
        df (pandas.DataFrame): The DataFrame to analyze.
    """
    msno.bar(df)
    plt.show()
