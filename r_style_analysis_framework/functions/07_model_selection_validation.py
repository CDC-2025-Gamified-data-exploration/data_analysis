# Category 7: Model Selection & Validation
# This module provides functions for comparing models and validating their performance.

# --- Information Criteria ---

def get_aic(model_results):
    """
    Equivalent to R's AIC()
    Uses the .aic attribute of a fitted statsmodels object.
    """
    return model_results.aic

def get_bic(model_results):
    """
    Equivalent to R's BIC()
    Uses the .bic attribute of a fitted statsmodels object.
    """
    return model_results.bic

def compare_models_aic_bic(models_dict):
    """
    Equivalent to R's MuMIn::model.sel()
    - Takes a dictionary of fitted models and returns a DataFrame with their AIC/BIC.
    - Libraries: pandas
    """
    pass

# --- Cross-Validation ---

def perform_k_fold_cv(model, X, y, k=10, scoring='neg_mean_squared_error'):
    """
    Equivalent to custom k-fold CV or using libraries like caret in R.
    - Uses sklearn.model_selection.cross_val_score
    - `model` should be a scikit-learn compatible estimator.
    - Libraries: scikit-learn
    """
    pass

def perform_loocv(model, X, y, scoring='neg_mean_squared_error'):
    """
    Performs Leave-One-Out Cross-Validation.
    - Uses sklearn.model_selection.LeaveOneOut
    - Libraries: scikit-learn
    """
    pass

# --- Bootstrap ---

def perform_bootstrap(data, n_iterations=1000, func=np.mean):
    """
    Equivalent to R's boot::boot()
    - A generic bootstrap resampling function.
    - `data` is an array-like object.
    - `func` is the statistic to calculate on each bootstrap sample.
    - Libraries: numpy, scikit-learn.utils
    """
    pass

def get_bootstrap_ci(data, n_iterations=1000, alpha=0.05, func=np.mean):
    """
    Equivalent to R's boot::boot.ci()
    - Calculates bootstrap confidence intervals.
    """
    pass

# --- Model Comparison ---

def perform_likelihood_ratio_test(model1, model2):
    """
    Equivalent to R's lmtest::lrtest()
    - Compares two nested models.
    - Uses statsmodels.stats.weightstats.CompareMeans.lr_test() or similar.
    - Libraries: statsmodels
    """
    pass

def compare_nested_models_anova(model1, model2):
    """
    Equivalent to R's anova(model1, model2)
    Uses statsmodels.stats.anova.anova_lm()
    - Libraries: statsmodels
    """
    pass
