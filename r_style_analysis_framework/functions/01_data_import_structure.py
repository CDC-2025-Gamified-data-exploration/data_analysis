# Category 1: Data Import & Structure
# This module maps common R functions for data import and inspection to their Python equivalents.

import pandas as pd

# --- Data Reading ---

def read_csv(filepath, **kwargs):
    """
    Equivalent to R's read.csv()
    Uses pandas.read_csv()
    """
    return pd.read_csv(filepath, **kwargs)

def read_excel(filepath, **kwargs):
    """
    Equivalent to R's readxl::read_excel()
    Uses pandas.read_excel()
    """
    return pd.read_excel(filepath, **kwargs)

def read_delim(filepath, **kwargs):
    """
    Equivalent to R's read.delim()
    Uses pandas.read_csv() with a specified separator.
    """
    return pd.read_csv(filepath, **kwargs)

# --- Data Inspection ---

def get_structure(df):
    """
    Equivalent to R's str() or dplyr::glimpse()
    Uses pandas.DataFrame.info()
    """
    return df.info()

def head(df, n=5):
    """
    Equivalent to R's head()
    Uses pandas.DataFrame.head()
    """
    return df.head(n)

def tail(df, n=5):
    """
    Equivalent to R's tail()
    Uses pandas.DataFrame.tail()
    """
    return df.tail(n)

def dim(df):
    """
    Equivalent to R's dim()
    Uses pandas.DataFrame.shape
    """
    return df.shape

def names(df):
    """
    Equivalent to R's names()
    Uses pandas.DataFrame.columns
    """
    return df.columns

def summary(df):
    """
    Equivalent to R's summary()
    Uses pandas.DataFrame.describe()
    """
    return df.describe(include='all')

# --- Data Types ---

def check_is_na(series):
    """
    Equivalent to R's is.na()
    Uses pandas.Series.isna()
    """
    return series.isna()

def get_complete_cases(df):
    """
    Equivalent to R's complete.cases()
    Uses pandas.DataFrame.dropna()
    """
    return df.dropna()

def change_dtype(series, dtype):
    """
    Equivalent to R's as.numeric(), as.factor(), etc.
    Uses pandas.Series.astype()
    """
    return series.astype(dtype)
