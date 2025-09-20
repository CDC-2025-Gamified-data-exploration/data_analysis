# This file contains the function for calculating a correlation matrix.

def get_correlation_matrix(df, method='pearson'):
    """
    Computes the pairwise correlation of columns in a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame with numerical columns.
        method (str): The method of correlation ('pearson', 'spearman', 'kendall').

    Returns:
        pandas.DataFrame: The correlation matrix.
    """
    return df.corr(method=method)
