# This file contains the function for detecting outliers using the Isolation Forest algorithm.

from sklearn.ensemble import IsolationForest

def detect_outliers_isolation_forest(df, contamination=0.1):
    """
    Detects outliers using the Isolation Forest algorithm.

    Args:
        df (pandas.DataFrame): The DataFrame to analyze.
        contamination (float): The expected proportion of outliers in the data.

    Returns:
        numpy.ndarray: An array of predictions (-1 for outliers, 1 for inliers).
    """
    model = IsolationForest(contamination=contamination, random_state=0)
    return model.fit_predict(df)
