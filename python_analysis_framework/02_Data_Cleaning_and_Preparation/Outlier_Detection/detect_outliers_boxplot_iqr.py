# This file contains the function for detecting outliers using the IQR method.

import numpy as np

def detect_outliers_iqr(series, factor=1.5):
    """
    Detects outliers in a Series using the Interquartile Range (IQR) method.

    Args:
        series (pandas.Series): The Series to analyze.
        factor (float): The IQR factor to use for the fences.

    Returns:
        pandas.Series: A boolean Series indicating the outliers.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (factor * iqr)
    upper_bound = q3 + (factor * iqr)

    return (series < lower_bound) | (series > upper_bound)
