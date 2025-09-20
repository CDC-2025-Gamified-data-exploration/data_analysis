# This file contains the function for calculating a wide range of summary statistics.

import pandas as pd

def get_summary_statistics(series):
    """
    Calculates a comprehensive set of summary statistics for a Series.

    Args:
        series (pandas.Series): The data series to analyze.

    Returns:
        pandas.Series: A Series containing the summary statistics.
    """
    stats = {
        'mean': series.mean(),
        'median': series.median(),
        'std_dev': series.std(),
        'variance': series.var(),
        'range': series.max() - series.min(),
        'iqr': series.quantile(0.75) - series.quantile(0.25),
        'q1': series.quantile(0.25),
        'q3': series.quantile(0.75),
        'skewness': series.skew(),
        'kurtosis': series.kurtosis()
    }
    return pd.Series(stats, name=series.name)
