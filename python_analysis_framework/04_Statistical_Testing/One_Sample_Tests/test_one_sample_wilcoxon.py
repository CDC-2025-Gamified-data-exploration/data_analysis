# This file contains the function for the one-sample Wilcoxon signed-rank test.

from scipy import stats

def test_one_sample_wilcoxon(series, y=None, alternative='two-sided'):
    """
    Performs the Wilcoxon signed-rank test for one sample.

    Args:
        series (array-like): The data sample.
        y (array-like, optional): A second sample for a paired test, or None for a one-sample test.
        alternative (str): Defines the alternative hypothesis ('two-sided', 'less', 'greater').

    Returns:
        tuple: A tuple containing the test statistic and the p-value.
    """
    return stats.wilcoxon(series, y=y, alternative=alternative)
