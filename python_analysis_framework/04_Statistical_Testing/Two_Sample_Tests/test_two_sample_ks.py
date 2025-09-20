# This file contains the function for the two-sample Kolmogorov-Smirnov test.

from scipy import stats

def test_two_sample_ks(sample1, sample2):
    """
    Performs the two-sample Kolmogorov-Smirnov test to check if two samples
    come from the same distribution.

    Args:
        sample1 (array-like): The first sample.
        sample2 (array-like): The second sample.

    Returns:
        tuple: A tuple containing the test statistic and the p-value.
    """
    return stats.ks_2samp(sample1, sample2)
