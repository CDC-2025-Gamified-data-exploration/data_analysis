# This file contains the function for detecting outliers using the Local Outlier Factor (LOF).

from sklearn.neighbors import LocalOutlierFactor

def detect_outliers_lof(df, n_neighbors=20):
    """
    Detects outliers using the Local Outlier Factor (LOF) algorithm.

    Args:
        df (pandas.DataFrame): The DataFrame to analyze.
        n_neighbors (int): Number of neighbors to use.

    Returns:
        numpy.ndarray: An array of predictions (-1 for outliers, 1 for inliers).
    """
    model = LocalOutlierFactor(n_neighbors=n_neighbors)
    return model.fit_predict(df)
