# This file contains the function for the exact binomial test.

from scipy import stats

def test_one_sample_binomial(count, nobs, p=0.5):
    """
    Performs an exact binomial test for a one-sample proportion.

    Args:
        count (int): The number of successes.
        nobs (int): The total number of trials.
        p (float): The hypothesized probability of success.

    Returns:
        float: The p-value of the test.
    """
    return stats.binomtest(count, nobs, p).pvalue
