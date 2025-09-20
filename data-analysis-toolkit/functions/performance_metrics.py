# This file will contain functions for calculating model performance metrics.

# --- Regression Metrics ---
def calculate_rmse(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error (RMSE).
    - Input: true values, predicted values
    - Output: RMSE score
    - Libraries: sklearn.metrics
    """
    pass

def calculate_mae(y_true, y_pred):
    """
    Calculates the Mean Absolute Error (MAE).
    - Input: true values, predicted values
    - Output: MAE score
    - Libraries: sklearn.metrics
    """
    pass

def calculate_mape(y_true, y_pred):
    """
    Calculates the Mean Absolute Percentage Error (MAPE).
    - Input: true values, predicted values
    - Output: MAPE score
    - Libraries: numpy
    """
    pass

def calculate_r_squared(y_true, y_pred):
    """
    Calculates the R-squared (coefficient of determination).
    - Input: true values, predicted values
    - Output: R-squared score
    - Libraries: sklearn.metrics
    """
    pass

# --- Classification Metrics ---
def plot_confusion_matrix(y_true, y_pred):
    """
    Calculates and plots a confusion matrix.
    - Input: true labels, predicted labels
    - Output: matplotlib plot object
    - Libraries: sklearn.metrics, seaborn
    """
    pass

def plot_roc_curve(y_true, y_pred_proba):
    """
    Calculates and plots the Receiver Operating Characteristic (ROC) curve and AUC.
    - Input: true labels, predicted probabilities for the positive class
    - Output: matplotlib plot object
    - Libraries: sklearn.metrics, matplotlib
    """
    pass

def plot_precision_recall_curve(y_true, y_pred_proba):
    """
    Calculates and plots the Precision-Recall curve.
    - Input: true labels, predicted probabilities for the positive class
    - Output: matplotlib plot object
    - Libraries: sklearn.metrics, matplotlib
    """
    pass

def calculate_classification_report(y_true, y_pred):
    """
    Generates a text report showing the main classification metrics (precision, recall, F1-score).
    - Input: true labels, predicted labels
    - Output: string report
    - Libraries: sklearn.metrics
    """
    pass

def calculate_cohens_kappa(y_true, y_pred):
    """
    Calculates Cohen's Kappa score.
    - Measures inter-annotator agreement.
    - Input: true labels, predicted labels
    - Output: kappa score
    - Libraries: sklearn.metrics
    """
    pass
