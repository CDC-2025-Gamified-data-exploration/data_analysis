# This file contains the function for the Yeo-Johnson transformation.

from scipy import stats

def transform_yeo_johnson(series):
    """
    Performs a Yeo-Johnson transformation to normalize a Series.
    This transformation can handle non-positive data.

    Args:
        series (pandas.Series): The Series to transform.

    Returns:
        tuple: A tuple containing the transformed Series and the optimal lambda value.
    """
    transformed_data, optimal_lambda = stats.yeojohnson(series)
    return transformed_data, optimal_lambda
