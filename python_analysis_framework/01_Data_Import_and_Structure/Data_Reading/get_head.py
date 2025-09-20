# This file contains the function for viewing the first few rows of a DataFrame.

def get_head(df, n=5):
    """
    Returns the first n rows of a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to view.
        n (int): The number of rows to return.

    Returns:
        pandas.DataFrame: The first n rows.
    """
    return df.head(n)
