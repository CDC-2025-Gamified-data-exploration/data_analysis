# This file contains the function for detecting multivariate outliers using Mahalanobis distance.

import numpy as np
from scipy.spatial import distance

def detect_outliers_mahalanobis(df):
    """
    Calculates the Mahalanobis distance for each row in a DataFrame to detect multivariate outliers.

    Args:
        df (pandas.DataFrame): The DataFrame with numerical features to analyze.

    Returns:
        numpy.ndarray: An array of Mahalanobis distances.
    """
    # Note: Outlier detection requires choosing a threshold, often from a chi-squared distribution.
    # This function just returns the distances.

    cov_matrix = np.cov(df.values.T)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    mean = np.mean(df, axis=0)

    distances = [distance.mahalanobis(row, mean, inv_cov_matrix) for index, row in df.iterrows()]
    return np.array(distances)
