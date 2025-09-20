# This file contains the function for creating a contingency table.

import pandas as pd

def get_contingency_table(series1, series2):
    """
    Creates a contingency table (cross-tabulation) of two categorical Series.

    Args:
        series1 (pandas.Series): The first categorical series.
        series2 (pandas.Series): The second categorical series.

    Returns:
        pandas.DataFrame: The contingency table.
    """
    return pd.crosstab(series1, series2)
