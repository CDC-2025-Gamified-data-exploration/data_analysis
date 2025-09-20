# This file contains the function for getting the dimensions of a DataFrame.

def get_dim(df):
    """
    Returns the dimensions (rows, columns) of a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to inspect.

    Returns:
        tuple: A tuple containing the number of rows and columns.
    """
    return df.shape
