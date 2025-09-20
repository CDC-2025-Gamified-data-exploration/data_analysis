# This file contains functions for calculating skewness and kurtosis.

def get_skewness(series):
    """
    Calculates the skewness of a Series.
    """
    return series.skew()

def get_kurtosis(series):
    """
    Calculates the kurtosis of a Series.
    """
    return series.kurtosis()
