# This file will contain functions for performing two-sample statistical tests.

def test_independent_t(sample1, sample2, equal_var=True):
    """
    Performs an independent samples t-test.
    - Can handle equal or unequal variances (Welch's t-test).
    - Input: two array-like samples, boolean for equal variance
    - Output: t-statistic, p-value
    - Libraries: scipy.stats
    """
    pass

def test_mann_whitney_u(sample1, sample2):
    """
    Performs the Mann-Whitney U test (non-parametric alternative to independent t-test).
    - Input: two array-like samples
    - Output: U-statistic, p-value
    - Libraries: scipy.stats
    """
    pass

def test_kolmogorov_smirnov_2sample(sample1, sample2):
    """
    Performs the two-sample Kolmogorov-Smirnov test to check if two samples come from the same distribution.
    - Input: two array-like samples
    - Output: test statistic, p-value
    - Libraries: scipy.stats
    """
    pass

def test_chi_square_independence(contingency_table):
    """
    Performs the Chi-square test of independence for two categorical variables.
    - Input: pandas DataFrame or numpy array (contingency table)
    - Output: chi2 statistic, p-value, degrees of freedom, expected frequencies
    - Libraries: scipy.stats
    """
    pass

def test_fishers_exact(contingency_table):
    """
    Performs Fisher's exact test, useful for small sample sizes in contingency tables.
    - Input: 2x2 contingency table (array-like)
    - Output: odds ratio, p-value
    - Libraries: scipy.stats
    """
    pass

def test_two_sample_proportion(count, nobs, alternative='two-sided'):
    """
    Performs a test for the equality of two proportions.
    - Input: list/tuple of counts, list/tuple of sample sizes
    - Output: z-statistic, p-value
    - Libraries: statsmodels.stats.proportion
    """
    pass
