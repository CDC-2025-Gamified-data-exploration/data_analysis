# This file contains the function for the Chi-square test of independence.

from scipy.stats import chi2_contingency

def test_chi_square_independence(contingency_table):
    """
    Performs the Chi-square test of independence on a contingency table.

    Args:
        contingency_table (pandas.DataFrame or array-like): The contingency table.

    Returns:
        tuple: A tuple containing the chi2 statistic, p-value, degrees of freedom,
               and expected frequencies.
    """
    return chi2_contingency(contingency_table)
