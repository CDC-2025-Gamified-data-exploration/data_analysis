# This file contains the function for calculating association statistics.

import numpy as np
from scipy.stats import chi2_contingency

def get_association_stats(contingency_table):
    """
    Calculates association statistics for a contingency table, such as Cramer's V.

    Args:
        contingency_table (pandas.DataFrame or array-like): The contingency table.

    Returns:
        dict: A dictionary of association statistics.
    """
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    cramers_v = np.sqrt(phi2 / min((k-1), (r-1)))

    return {'cramers_v': cramers_v}
