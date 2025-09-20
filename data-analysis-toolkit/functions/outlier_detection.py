# This file will contain functions for detecting and treating outliers.

# --- Statistical Methods for Outlier Detection ---
def detect_outliers_z_score(df, column, threshold=3):
    """
    Detects outliers using the Z-score method.
    - Input: DataFrame, column name, Z-score threshold
    - Output: DataFrame with a boolean column 'is_outlier'
    - Libraries: pandas, numpy
    """
    pass

def detect_outliers_iqr(df, column, factor=1.5):
    """
    Detects outliers using the Interquartile Range (IQR) method.
    - Input: DataFrame, column name, IQR factor
    - Output: DataFrame with a boolean column 'is_outlier'
    - Libraries: pandas, numpy
    """
    pass

def detect_outliers_modified_z_score(df, column, threshold=3.5):
    """
    Detects outliers using the Modified Z-score (using median absolute deviation).
    - Input: DataFrame, column name, threshold
    - Output: DataFrame with a boolean column 'is_outlier'
    - Libraries: pandas, numpy, statsmodels
    """
    pass

# --- Multivariate Outlier Detection ---
def detect_outliers_mahalanobis(df, columns):
    """
    Detects multivariate outliers using Mahalanobis distance.
    - Input: DataFrame, list of column names
    - Output: DataFrame with Mahalanobis distance and outlier flag
    - Libraries: numpy, scipy
    """
    pass

def detect_outliers_isolation_forest(df, columns, contamination=0.1):
    """
    Detects outliers using the Isolation Forest algorithm.
    - Input: DataFrame, list of column names, expected contamination
    - Output: DataFrame with an 'is_outlier' column (-1 for outliers, 1 for inliers)
    - Libraries: scikit-learn
    """
    pass

def detect_outliers_lof(df, columns, n_neighbors=20):
    """
    Detects outliers using the Local Outlier Factor (LOF) algorithm.
    - Input: DataFrame, list of column names, number of neighbors
    - Output: DataFrame with an 'is_outlier' column (-1 for outliers, 1 for inliers)
    - Libraries: scikit-learn
    """
    pass

# --- Visualization Methods ---
def plot_boxplot_outliers(df, column):
    """
    Visualizes outliers using a boxplot.
    - Input: DataFrame, column name
    - Output: matplotlib plot object
    - Libraries: matplotlib, seaborn
    """
    pass

def plot_scatterplot_outliers(df, x_col, y_col):
    """
    Visualizes outliers on a scatter plot, highlighting them in a different color.
    Assumes outlier detection has already been run and an 'is_outlier' column exists.
    - Input: DataFrame, x-axis column, y-axis column
    - Output: matplotlib plot object
    - Libraries: matplotlib, seaborn
    """
    pass
