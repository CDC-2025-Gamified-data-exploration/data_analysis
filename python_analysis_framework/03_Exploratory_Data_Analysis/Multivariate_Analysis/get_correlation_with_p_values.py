# This file contains the function for calculating a correlation matrix with p-values.

from scipy.stats import pearsonr
import pandas as pd

def get_correlation_with_p_values(df):
    """
    Calculates a correlation matrix and a corresponding matrix of p-values.

    Args:
        df (pandas.DataFrame): The DataFrame with numerical columns.

    Returns:
        tuple: A tuple containing two DataFrames: the correlation matrix and the p-value matrix.
    """
    df = df.dropna()._get_numeric_data()
    cols = pd.DataFrame(columns=df.columns)
    p_values = cols.transpose().join(cols, how='outer')

    for r in df.columns:
        for c in df.columns:
            p_values[r][c] = round(pearsonr(df[r], df[c])[1], 4)

    corr_matrix = df.corr()
    return corr_matrix, p_values
