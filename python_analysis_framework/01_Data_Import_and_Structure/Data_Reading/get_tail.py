# This file contains the function for viewing the last few rows of a DataFrame.

def get_tail(df, n=5):
    """
    Returns the last n rows of a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to view.
        n (int): The number of rows to return.

    Returns:
        pandas.DataFrame: The last n rows.
    """
    return df.tail(n)
