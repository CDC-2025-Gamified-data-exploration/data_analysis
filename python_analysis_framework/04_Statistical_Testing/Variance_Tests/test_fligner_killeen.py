# This file contains the function for the Fligner-Killeen test for equal variances.

from scipy import stats

def test_fligner_killeen(*samples):
    """
    Performs the Fligner-Killeen test to assess if samples have equal variances.
    This is a non-parametric test that is very robust against departures from normality.

    Args:
        *samples: Two or more array-like data samples.

    Returns:
        tuple: A tuple containing the test statistic and the p-value.
    """
    return stats.fligner(*samples)
