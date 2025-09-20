# This file contains the function for getting the column names of a DataFrame.

def get_names(df):
    """
    Returns the column labels of the DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to inspect.

    Returns:
        pandas.Index: The column labels.
    """
    return df.columns
