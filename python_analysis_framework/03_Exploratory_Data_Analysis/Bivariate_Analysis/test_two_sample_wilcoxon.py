# This file contains the function for performing a two-sample Wilcoxon test (Mann-Whitney U).

from scipy import stats

def test_two_sample_wilcoxon(sample1, sample2):
    """
    Performs the Mann-Whitney U rank test on two independent samples.

    Args:
        sample1 (array-like): The first sample.
        sample2 (array-like): The second sample.

    Returns:
        tuple: A tuple containing the U-statistic and the p-value.
    """
    return stats.mannwhitneyu(sample1, sample2)
