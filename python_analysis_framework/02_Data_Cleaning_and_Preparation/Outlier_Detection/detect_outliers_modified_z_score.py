# This file contains the function for detecting outliers using the Modified Z-score.

import numpy as np

def detect_outliers_modified_z_score(series, threshold=3.5):
    """
    Detects outliers using the Modified Z-score, which is based on the Median Absolute Deviation (MAD).

    Args:
        series (pandas.Series): The Series to analyze.
        threshold (float): The threshold to use for identifying outliers.

    Returns:
        pandas.Series: A boolean Series indicating the outliers.
    """
    median = series.median()
    mad = (np.abs(series - median)).median()
    modified_z_scores = 0.6745 * (series - median) / mad

    return np.abs(modified_z_scores) > threshold
