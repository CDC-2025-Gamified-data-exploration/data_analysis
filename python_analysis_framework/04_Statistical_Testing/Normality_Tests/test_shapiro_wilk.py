# This file contains the function for the Shapiro-Wilk test for normality.

from scipy import stats

def test_shapiro_wilk(series):
    """
    Performs the Shapiro-Wilk test to assess if a sample comes from a normal distribution.

    Args:
        series (array-like): The data sample.

    Returns:
        tuple: A tuple containing the test statistic and the p-value.
    """
    return stats.shapiro(series)
