# This file contains the function for robust scaling.

from sklearn.preprocessing import RobustScaler

def scale_robust(df):
    """
    Scales features using statistics that are robust to outliers.
    This method removes the median and scales the data according to the quantile range.

    Args:
        df (pandas.DataFrame): The DataFrame with numerical features to scale.

    Returns:
        numpy.ndarray: The scaled data.
    """
    scaler = RobustScaler()
    return scaler.fit_transform(df)
