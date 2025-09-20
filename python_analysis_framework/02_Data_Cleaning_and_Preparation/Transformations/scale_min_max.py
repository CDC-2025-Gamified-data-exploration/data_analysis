# This file contains the function for Min-Max scaling (normalization).

from sklearn.preprocessing import MinMaxScaler

def scale_min_max(df):
    """
    Transforms features by scaling each feature to a given range, typically [0, 1].

    Args:
        df (pandas.DataFrame): The DataFrame with numerical features to scale.

    Returns:
        numpy.ndarray: The scaled data.
    """
    scaler = MinMaxScaler()
    return scaler.fit_transform(df)
