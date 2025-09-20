# This file contains the function for the Box-Cox transformation.

from scipy import stats

def transform_box_cox(series):
    """
    Performs a Box-Cox transformation to normalize a Series.
    Note: The data must be positive.

    Args:
        series (pandas.Series): The Series to transform.

    Returns:
        tuple: A tuple containing the transformed Series and the optimal lambda value.
    """
    transformed_data, optimal_lambda = stats.boxcox(series)
    return transformed_data, optimal_lambda
