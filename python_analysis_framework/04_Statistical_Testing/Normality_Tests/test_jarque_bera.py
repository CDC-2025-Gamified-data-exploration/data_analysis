# This file contains the function for the Jarque-Bera test for normality.

from scipy import stats

def test_jarque_bera(series):
    """
    Performs the Jarque-Bera test to assess if a sample has the skewness and kurtosis
    matching a normal distribution.

    Args:
        series (array-like): The data sample.

    Returns:
        tuple: A tuple containing the test statistic and the p-value.
    """
    return stats.jarque_bera(series)
