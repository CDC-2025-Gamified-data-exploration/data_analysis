# This file contains the function for the Kolmogorov-Smirnov test for goodness of fit.

from scipy import stats

def test_kolmogorov_smirnov(series, distribution='norm'):
    """
    Performs the Kolmogorov-Smirnov test to assess if a sample comes from a specified distribution.

    Args:
        series (array-like): The data sample.
        distribution (str): The name of the distribution to test against (e.g., 'norm', 'expon').

    Returns:
        tuple: A tuple containing the test statistic and the p-value.
    """
    return stats.kstest(series, distribution)
