# This file will contain functions for model selection criteria.

def calculate_aic(model_results):
    """
    Calculates the Akaike Information Criterion (AIC) for a model.
    - Input: statsmodels results object
    - Output: AIC value
    - Libraries: statsmodels
    """
    pass

def calculate_bic(model_results):
    """
    Calculates the Bayesian Information Criterion (BIC) for a model.
    - Input: statsmodels results object
    - Output: BIC value
    - Libraries: statsmodels
    """
    pass

def calculate_mallows_cp(model_results, full_model_results):
    """
    Calculates Mallows' Cp statistic for a subset model.
    - Requires a fitted full model for comparison.
    - Input: results object for the subset model, results object for the full model
    - Output: Mallows' Cp value
    - Libraries: (custom implementation using model results)
    """
    pass

def compare_models_information_criteria(models):
    """
    Creates a table comparing AIC and BIC for a list of fitted models.
    - Input: list of statsmodels results objects
    - Output: pandas DataFrame with model comparison
    - Libraries: pandas
    """
    pass

def plot_mallows_cp(results_df):
    """
    Plots Mallows' Cp vs. the number of parameters for best subset regression results.
    - Helps to visually select the best model.
    - Input: DataFrame containing Cp values and number of parameters for various models
    - Output: matplotlib plot object
    - Libraries: matplotlib
    """
    pass

def calculate_adjusted_r_squared(model_results):
    """
    Calculates the Adjusted R-squared for a model.
    - Input: statsmodels results object
    - Output: Adjusted R-squared value
    - Libraries: statsmodels
    """
    pass
