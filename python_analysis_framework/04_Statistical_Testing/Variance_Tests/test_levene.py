# This file contains the function for Levene's test for equal variances.

from scipy import stats

def test_levene(*samples):
    """
    Performs Levene's test to assess if samples have equal variances.
    This test is less sensitive to departures from normality than Bartlett's test.

    Args:
        *samples: Two or more array-like data samples.

    Returns:
        tuple: A tuple containing the test statistic and the p-value.
    """
    return stats.levene(*samples)
