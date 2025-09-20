# This file will contain functions for formally testing regression assumptions.

# --- Linearity ---
def test_linearity_reset(model_results):
    """
    Performs Ramsey's RESET test for functional form (linearity).
    - Null hypothesis: the model is linear.
    - Input: statsmodels results object
    - Output: F-statistic, p-value
    - Libraries: statsmodels.stats.diagnostic
    """
    pass

def test_linearity_rainbow(model_results):
    """
    Performs the Rainbow test for linearity.
    - Null hypothesis: the model is linear.
    - Input: statsmodels results object
    - Output: F-statistic, p-value
    - Libraries: statsmodels.stats.diagnostic
    """
    pass

# --- Independence of Residuals ---
def test_durbin_watson(model_results):
    """
    Performs the Durbin-Watson test for autocorrelation in the residuals.
    - Test statistic is approximately 2 for no autocorrelation.
    - Input: statsmodels results object
    - Output: Durbin-Watson statistic
    - Libraries: statsmodels.stats.stattools
    """
    pass

def test_ljung_box(model_results, lags=None):
    """
    Performs the Ljung-Box test for autocorrelation.
    - Input: statsmodels results object, number of lags to test
    - Output: test statistic, p-value
    - Libraries: statsmodels.stats.diagnostic
    """
    pass

def plot_acf_pacf_residuals(model_results, lags=None):
    """
    Plots the ACF and PACF of the residuals to visually check for autocorrelation.
    - Input: statsmodels results object, number of lags
    - Output: matplotlib figure object
    - Libraries: statsmodels.graphics.tsaplots
    """
    pass

# --- Homoscedasticity (Constant Variance) ---
def test_breusch_pagan(model_results):
    """
    Performs the Breusch-Pagan test for heteroscedasticity.
    - Null hypothesis: homoscedasticity is present.
    - Input: statsmodels results object
    - Output: Lagrange multiplier statistic, p-value
    - Libraries: statsmodels.stats.diagnostic
    """
    pass

def test_white(model_results):
    """
    Performs White's test for heteroscedasticity.
    - More general than Breusch-Pagan.
    - Input: statsmodels results object
    - Output: Lagrange multiplier statistic, p-value
    - Libraries: statsmodels.stats.diagnostic
    """
    pass

# --- Normality of Residuals ---
def test_shapiro_on_residuals(model_results):
    """
    Performs the Shapiro-Wilk test on the model's residuals.
    - Input: statsmodels results object
    - Output: test statistic, p-value
    - Libraries: scipy.stats
    """
    pass

def test_jarque_bera_on_residuals(model_results):
    """
    Performs the Jarque-Bera test on the model's residuals.
    - This is often included in the default statsmodels summary.
    - Input: statsmodels results object
    - Output: JB statistic, p-value, skew, kurtosis
    - Libraries: statsmodels.stats.stattools
    """
    pass
