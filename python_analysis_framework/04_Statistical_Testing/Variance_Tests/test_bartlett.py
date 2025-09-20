# This file contains the function for Bartlett's test for equal variances.

from scipy import stats

def test_bartlett(*samples):
    """
    Performs Bartlett's test to assess if samples have equal variances.
    Note: This test is sensitive to the assumption of normality.

    Args:
        *samples: Two or more array-like data samples.

    Returns:
        tuple: A tuple containing the test statistic and the p-value.
    """
    return stats.bartlett(*samples)
