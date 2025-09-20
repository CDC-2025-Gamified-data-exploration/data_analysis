# This file contains the function for performing a Chi-square goodness of fit test.

from scipy.stats import chisquare

def test_goodness_of_fit(observed_counts, expected_counts=None):
    """
    Performs a Chi-square goodness of fit test.
    This test is used to determine whether a categorical variable has a given frequency distribution.

    Args:
        observed_counts (array-like): The observed frequencies in each category.
        expected_counts (array-like, optional): The expected frequencies in each category.
                                                If not given, assumes a uniform distribution.

    Returns:
        tuple: A tuple containing the chi-squared statistic and the p-value.
    """
    return chisquare(f_obs=observed_counts, f_exp=expected_counts)
