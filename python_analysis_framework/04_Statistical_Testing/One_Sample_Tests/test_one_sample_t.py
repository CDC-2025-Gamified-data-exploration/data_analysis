# This file contains the function for the one-sample t-test.

from scipy import stats

def test_one_sample_t(series, popmean):
    """
    Performs a one-sample t-test to check if the mean of a sample is equal to a known value.

    Args:
        series (array-like): The data sample.
        popmean (float): The population mean to test against.

    Returns:
        tuple: A tuple containing the t-statistic and the p-value.
    """
    return stats.ttest_1samp(series, popmean)
