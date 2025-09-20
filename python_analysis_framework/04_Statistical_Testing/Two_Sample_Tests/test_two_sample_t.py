# This file contains the function for the independent two-sample t-test.

from scipy import stats

def test_two_sample_t(sample1, sample2, equal_var=True):
    """
    Performs an independent two-sample t-test.

    Args:
        sample1 (array-like): The first sample.
        sample2 (array-like): The second sample.
        equal_var (bool): If True, performs a standard independent t-test.
                          If False, performs Welch's t-test.

    Returns:
        tuple: A tuple containing the t-statistic and the p-value.
    """
    return stats.ttest_ind(sample1, sample2, equal_var=equal_var)
