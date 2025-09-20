# This file will contain functions for performing and visualizing normality tests.

def test_shapiro_wilk(series):
    """
    Performs the Shapiro-Wilk test for normality.
    - Input: pandas Series
    - Output: test statistic, p-value
    - Libraries: scipy.stats
    """
    pass

def test_kolmogorov_smirnov(series):
    """
    Performs the Kolmogorov-Smirnov test for normality.
    - Input: pandas Series
    - Output: test statistic, p-value
    - Libraries: scipy.stats
    """
    pass

def test_anderson_darling(series):
    """
    Performs the Anderson-Darling test for normality.
    - Input: pandas Series
    - Output: test statistic, critical values, significance levels
    - Libraries: scipy.stats
    """
    pass

def test_jarque_bera(series):
    """
    Performs the Jarque-Bera test for normality.
    - Input: pandas Series
    - Output: test statistic, p-value
    - Libraries: scipy.stats
    """
    pass

def test_lilliefors(series):
    """
    Performs the Lilliefors test for normality (a correction of the K-S test).
    - Input: pandas Series
    - Output: test statistic, p-value
    - Libraries: statsmodels.stats.diagnostic
    """
    pass

def run_all_normality_tests(series):
    """
    Runs a suite of normality tests on a series and returns a summary DataFrame.
    - Input: pandas Series
    - Output: pandas DataFrame with test results
    - Libraries: pandas, scipy.stats, statsmodels
    """
    pass

def plot_normality_dashboard(series):
    """
    Creates a dashboard of visual normality checks.
    - Includes a histogram with normal curve overlay and a Q-Q plot.
    - Input: pandas Series
    - Output: matplotlib figure object with subplots
    - Libraries: matplotlib, seaborn, statsmodels.api, scipy.stats
    """
    pass
