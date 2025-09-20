# This file will contain functions for data transformation.

# --- Normality Transformations ---
def transform_log(df, column):
    """
    Applies a log transformation to a column.
    - Input: DataFrame, column name
    - Output: Transformed Series/DataFrame
    - Libraries: numpy
    """
    pass

def transform_sqrt(df, column):
    """
    Applies a square root transformation to a column.
    - Input: DataFrame, column name
    - Output: Transformed Series/DataFrame
    - Libraries: numpy
    """
    pass

def transform_box_cox(df, column):
    """
    Applies a Box-Cox transformation to a column.
    - Input: DataFrame, column name
    - Output: Transformed Series/DataFrame and the lambda value
    - Libraries: scipy.stats
    """
    pass

def transform_yeo_johnson(df, column):
    """
    Applies a Yeo-Johnson transformation to a column.
    - Input: DataFrame, column name
    - Output: Transformed Series/DataFrame and the lambda value
    - Libraries: scipy.stats
    """
    pass

# --- Scaling ---
def scale_min_max(df, columns):
    """
    Applies Min-Max normalization to specified columns.
    - Input: DataFrame, list of column names
    - Output: DataFrame with scaled columns
    - Libraries: scikit-learn (MinMaxScaler)
    """
    pass

def scale_z_score(df, columns):
    """
    Applies Z-score standardization to specified columns.
    - Input: DataFrame, list of column names
    - Output: DataFrame with scaled columns
    - Libraries: scikit-learn (StandardScaler)
    """
    pass

def scale_robust(df, columns):
    """
    Applies robust scaling (removes median, scales by IQR) to specified columns.
    - Input: DataFrame, list of column names
    - Output: DataFrame with scaled columns
    - Libraries: scikit-learn (RobustScaler)
    """
    pass

# --- Categorical Encoding ---
def encode_one_hot(df, column):
    """
    Applies one-hot encoding to a categorical column.
    - Input: DataFrame, column name
    - Output: DataFrame with new one-hot encoded columns
    - Libraries: pandas (get_dummies) or scikit-learn (OneHotEncoder)
    """
    pass

def encode_ordinal(df, column, category_order):
    """
    Applies ordinal encoding to a categorical column based on a specified order.
    - Input: DataFrame, column name, list defining the order of categories
    - Output: DataFrame with the encoded column
    - Libraries: pandas or scikit-learn (OrdinalEncoder)
    """
    pass

def encode_target(df, column, target_column):
    """
    Applies target encoding to a categorical column.
    - Input: DataFrame, column name, target column name
    - Output: DataFrame with the encoded column
    - Libraries: category_encoders (TargetEncoder)
    """
    pass

def encode_frequency(df, column):
    """
    Applies frequency encoding to a categorical column.
    - Input: DataFrame, column name
    - Output: DataFrame with the encoded column
    - Libraries: pandas
    """
    pass
