# This file contains the function for the F-test for equal variances.

import numpy as np
from scipy import stats

def test_f_test_for_variances(sample1, sample2):
    """
    Performs an F-test to compare the variances of two independent samples.
    Note: This test is sensitive to the assumption of normality.

    Args:
        sample1 (array-like): The first data sample.
        sample2 (array-like): The second data sample.

    Returns:
        tuple: A tuple containing the F-statistic and the p-value.
    """
    f_statistic = np.var(sample1, ddof=1) / np.var(sample2, ddof=1)
    df1 = len(sample1) - 1
    df2 = len(sample2) - 1
    p_value = 1 - stats.f.cdf(f_statistic, df1, df2)
    return f_statistic, p_value * 2 # Two-tailed test
