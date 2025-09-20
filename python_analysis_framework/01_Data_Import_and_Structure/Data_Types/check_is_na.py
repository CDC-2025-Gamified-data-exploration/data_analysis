# This file contains the function for checking for missing values.

def check_is_na(series):
    """
    Detects missing values in a pandas Series.

    Args:
        series (pandas.Series): The Series to check.

    Returns:
        pandas.Series: A boolean Series indicating missing values.
    """
    return series.isna()
