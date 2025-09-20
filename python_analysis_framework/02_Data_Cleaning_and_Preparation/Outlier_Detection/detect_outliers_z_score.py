# This file contains the function for detecting outliers using the Z-score method.

import numpy as np
from scipy import stats

def detect_outliers_z_score(series, threshold=3):
    """
    Detects outliers in a Series using the Z-score.

    Args:
        series (pandas.Series): The Series to analyze.
        threshold (float): The Z-score threshold to use for identifying outliers.

    Returns:
        pandas.Series: A boolean Series indicating the outliers.
    """
    z_scores = np.abs(stats.zscore(series))
    return z_scores > threshold
