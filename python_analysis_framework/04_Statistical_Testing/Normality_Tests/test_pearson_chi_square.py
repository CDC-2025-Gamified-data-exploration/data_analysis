# This file contains the function for the Pearson chi-square test for normality.

from scipy import stats
import numpy as np

def test_pearson_chi_square(series):
    """
    Performs the Pearson chi-square test for normality.
    Note: This test is generally less powerful for normality than tests like Shapiro-Wilk.
    It requires binning the data.

    Args:
        series (array-like): The data sample.

    Returns:
        tuple: A tuple containing the test statistic and the p-value.
    """
    # Create bins and get observed frequencies
    observed_freq, bins = np.histogram(series, bins='auto')

    # Get expected frequencies from a normal distribution
    mean, std = np.mean(series), np.std(series)
    expected_freq = [len(series) * (stats.norm.cdf(bins[i+1], mean, std) - stats.norm.cdf(bins[i], mean, std))
                     for i in range(len(bins)-1)]

    return stats.chisquare(f_obs=observed_freq, f_exp=expected_freq)
