# This file contains functions for simple mathematical transformations.

import numpy as np

def transform_log(series):
    """
    Applies a log transformation to a Series.
    """
    return np.log(series)

def transform_sqrt(series):
    """
    Applies a square root transformation to a Series.
    """
    return np.sqrt(series)

def transform_power(series, power):
    """
    Applies a power transformation to a Series.
    """
    return np.power(series, power)
