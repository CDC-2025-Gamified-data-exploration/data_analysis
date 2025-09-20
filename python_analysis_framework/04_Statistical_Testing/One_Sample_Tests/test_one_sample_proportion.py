# This file contains the function for the one-sample proportion test.

from statsmodels.stats.proportion import proportions_ztest

def test_one_sample_proportion(count, nobs, value=0.5):
    """
    Performs a one-sample proportion test to check if a sample proportion is equal to a known value.

    Args:
        count (int): The number of successes.
        nobs (int): The total number of trials.
        value (float): The hypothesized proportion.

    Returns:
        tuple: A tuple containing the z-statistic and the p-value.
    """
    return proportions_ztest(count, nobs, value)
