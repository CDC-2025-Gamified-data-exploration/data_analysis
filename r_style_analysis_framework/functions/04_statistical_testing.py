# Category 4: Statistical Testing
# This module provides a suite of functions for hypothesis testing.

# --- Normality Tests ---

def test_shapiro_wilk(series):
    """
    Equivalent to R's shapiro.test()
    Uses scipy.stats.shapiro()
    - Libraries: scipy.stats
    """
    pass

def test_kolmogorov_smirnov(series, dist='norm'):
    """
    Equivalent to R's ks.test()
    Uses scipy.stats.kstest()
    - Libraries: scipy.stats
    """
    pass

def test_anderson_darling(series):
    """
    Equivalent to R's nortest::ad.test()
    Uses scipy.stats.anderson()
    - Libraries: scipy.stats
    """
    pass

# --- Variance Tests ---

def test_levene(sample1, *samples):
    """
    Equivalent to R's car::leveneTest()
    Uses scipy.stats.levene()
    - Libraries: scipy.stats
    """
    pass

def test_bartlett(*samples):
    """
    Equivalent to R's bartlett.test()
    Uses scipy.stats.bartlett()
    - Libraries: scipy.stats
    """
    pass

# --- One Sample Tests ---

def test_one_sample_t(series, popmean):
    """
    Equivalent to R's t.test() for one sample
    Uses scipy.stats.ttest_1samp()
    - Libraries: scipy.stats
    """
    pass

def test_one_sample_wilcoxon(series, y=None, alternative='two-sided'):
    """
    Equivalent to R's wilcox.test() for one sample
    Uses scipy.stats.wilcoxon()
    - Libraries: scipy.stats
    """
    pass

# --- Two Sample Tests ---

def test_independent_t(sample1, sample2, equal_var=True):
    """
    Equivalent to R's t.test() for two independent samples
    Uses scipy.stats.ttest_ind()
    - Libraries: scipy.stats
    """
    pass

def test_mann_whitney_u(sample1, sample2):
    """
    Equivalent to R's wilcox.test() for two independent samples
    Uses scipy.stats.mannwhitneyu()
    - Libraries: scipy.stats
    """
    pass

# --- Multiple Sample Tests ---

def test_one_way_anova(*samples):
    """
    Equivalent to R's aov() for one-way ANOVA
    Uses scipy.stats.f_oneway()
    - Libraries: scipy.stats
    """
    pass

def test_kruskal_wallis(*samples):
    """
    Equivalent to R's kruskal.test()
    Uses scipy.stats.kruskal()
    - Libraries: scipy.stats
    """
    pass
