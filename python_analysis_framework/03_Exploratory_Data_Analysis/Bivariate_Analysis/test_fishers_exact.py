# This file contains the function for Fisher's exact test.

from scipy.stats import fisher_exact

def test_fishers_exact(contingency_table):
    """
    Performs Fisher's exact test on a 2x2 contingency table.

    Args:
        contingency_table (array-like): A 2x2 contingency table.

    Returns:
        tuple: A tuple containing the odds ratio and the p-value.
    """
    return fisher_exact(contingency_table)
