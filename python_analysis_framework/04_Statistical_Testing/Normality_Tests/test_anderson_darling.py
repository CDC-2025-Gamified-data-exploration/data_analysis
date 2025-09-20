# This file contains the function for the Anderson-Darling test for normality.

from scipy import stats

def test_anderson_darling(series):
    """
    Performs the Anderson-Darling test for assessing if a sample comes from a normal distribution.

    Args:
        series (array-like): The data sample.

    Returns:
        scipy.stats.AndersonResult: An object containing the test statistic, critical values,
                                    and significance levels.
    """
    return stats.anderson(series, dist='norm')
