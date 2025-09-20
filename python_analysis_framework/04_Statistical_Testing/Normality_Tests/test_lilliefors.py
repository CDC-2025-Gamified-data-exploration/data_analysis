# This file contains the function for the Lilliefors test for normality.

from statsmodels.stats.diagnostic import lilliefors

def test_lilliefors(series):
    """
    Performs the Lilliefors test, a correction of the Kolmogorov-Smirnov test for normality.

    Args:
        series (array-like): The data sample.

    Returns:
        tuple: A tuple containing the test statistic and the p-value.
    """
    return lilliefors(series)
