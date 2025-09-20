# This file contains the function for Z-score standardization.

from sklearn.preprocessing import StandardScaler

def scale_z_score(df):
    """
    Standardizes features by removing the mean and scaling to unit variance.

    Args:
        df (pandas.DataFrame): The DataFrame with numerical features to scale.

    Returns:
        numpy.ndarray: The scaled data.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(df)
