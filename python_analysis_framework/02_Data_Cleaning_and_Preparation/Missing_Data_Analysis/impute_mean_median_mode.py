# This file contains functions for simple mean, median, or mode imputation.

def impute_simple(series, method='mean'):
    """
    Imputes missing values in a Series using a simple strategy.

    Args:
        series (pandas.Series): The Series with missing data.
        method (str): The imputation method ('mean', 'median', or 'mode').

    Returns:
        pandas.Series: The imputed Series.
    """
    if method == 'mean':
        fill_value = series.mean()
    elif method == 'median':
        fill_value = series.median()
    elif method == 'mode':
        fill_value = series.mode()[0]
    else:
        raise ValueError("Method not recognized. Use 'mean', 'median', or 'mode'.")

    return series.fillna(fill_value)
