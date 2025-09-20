# This file contains the function for testing the significance of a correlation.

from scipy import stats

def test_correlation_significance(series1, series2, method='pearson'):
    """
    Tests the significance of the correlation between two Series.

    Args:
        series1 (pandas.Series): The first data series.
        series2 (pandas.Series): The second data series.
        method (str): The method of correlation ('pearson', 'spearman').

    Returns:
        tuple: A tuple containing the correlation coefficient and the p-value.
    """
    if method == 'pearson':
        return stats.pearsonr(series1, series2)
    elif method == 'spearman':
        return stats.spearmanr(series1, series2)
    else:
        raise ValueError("Method not recognized. Use 'pearson' or 'spearman'.")
