# This file contains the function for calculating correlation between two variables.

def calculate_correlation(series1, series2, method='pearson'):
    """
    Calculates the correlation between two Series.

    Args:
        series1 (pandas.Series): The first data series.
        series2 (pandas.Series): The second data series.
        method (str): The method of correlation ('pearson', 'spearman', 'kendall').

    Returns:
        float: The correlation coefficient.
    """
    return series1.corr(series2, method=method)
