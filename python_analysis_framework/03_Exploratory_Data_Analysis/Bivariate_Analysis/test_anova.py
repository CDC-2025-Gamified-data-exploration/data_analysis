# This file contains the function for performing a one-way ANOVA.

from scipy import stats

def test_anova(*samples):
    """
    Performs a one-way Analysis of Variance (ANOVA).

    Args:
        *samples: Two or more array-like objects representing the samples.

    Returns:
        tuple: A tuple containing the F-statistic and the p-value.
    """
    return stats.f_oneway(*samples)
