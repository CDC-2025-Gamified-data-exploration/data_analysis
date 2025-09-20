# This file will contain functions for performing one-sample statistical tests.

def test_one_sample_t(series, popmean):
    """
    Performs a one-sample t-test.
    - Checks assumptions like normality.
    - Input: pandas Series, population mean to test against
    - Output: t-statistic, p-value
    - Libraries: scipy.stats
    """
    pass

def test_wilcoxon_signed_rank(series, popmean):
    """
    Performs a Wilcoxon signed-rank test (non-parametric alternative to one-sample t-test).
    - Input: pandas Series, population median to test against
    - Output: test statistic, p-value
    - Libraries: scipy.stats
    """
    pass

def test_one_sample_proportion(count, nobs, p0):
    """
    Performs a one-sample proportion test.
    - Input: number of successes (count), total observations (nobs), hypothesized proportion (p0)
    - Output: z-statistic, p-value
    - Libraries: statsmodels.stats.proportion
    """
    pass

def calculate_bootstrap_ci(series, n_bootstrap=1000, alpha=0.05):
    """
    Calculates bootstrap confidence intervals for the mean of a sample.
    - Input: pandas Series, number of bootstrap samples, alpha level
    - Output: lower bound, upper bound of the confidence interval
    - Libraries: numpy
    """
    pass

def test_sign(series, popmedian):
    """
    Performs a sign test to determine if a sample median is equal to a hypothesized median.
    - Input: pandas Series, hypothesized median
    - Output: test statistic, p-value
    - Libraries: scipy.stats
    """
    pass
