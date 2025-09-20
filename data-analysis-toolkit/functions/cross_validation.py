# This file will contain functions for performing various cross-validation methods.

def perform_k_fold_cv(model, X, y, k=10):
    """
    Performs K-fold cross-validation.
    - Input: model object (e.g., from scikit-learn), feature matrix X, target vector y, number of folds k
    - Output: array of scores for each fold
    - Libraries: sklearn.model_selection
    """
    pass

def perform_loocv(model, X, y):
    """
    Performs Leave-One-Out Cross-Validation (LOOCV).
    - Input: model object, feature matrix X, target vector y
    - Output: array of scores for each fold
    - Libraries: sklearn.model_selection
    """
    pass

def perform_stratified_k_fold_cv(model, X, y, k=10):
    """
    Performs Stratified K-fold cross-validation, preserving class proportions.
    - Useful for classification tasks with imbalanced classes.
    - Input: model object, feature matrix X, target vector y, number of folds k
    - Output: array of scores for each fold
    - Libraries: sklearn.model_selection
    """
    pass

def perform_time_series_cv(model, X, y, n_splits=5):
    """
    Performs time series cross-validation (e.g., with a forward-chaining approach).
    - Input: model object, feature matrix X, target vector y, number of splits
    - Output: array of scores for each split
    - Libraries: sklearn.model_selection (TimeSeriesSplit)
    """
    pass

def perform_bootstrap_validation(model, X, y, n_iterations=100):
    """
    Performs validation using the bootstrap method (.632 bootstrap).
    - Input: model object, feature matrix X, target vector y, number of iterations
    - Output: array of scores
    - Libraries: (custom implementation or using sklearn utilities)
    """
    pass

def perform_monte_carlo_cv(model, X, y, n_iter=20, test_size=0.25):
    """
    Performs Monte Carlo cross-validation (repeated random sub-sampling).
    - Input: model object, X, y, number of iterations, test set size
    - Output: array of scores
    - Libraries: sklearn.model_selection (ShuffleSplit)
    """
    pass
