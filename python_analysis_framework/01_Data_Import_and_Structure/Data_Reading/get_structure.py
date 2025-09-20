# This file contains the function for inspecting the structure of a DataFrame.

def get_structure(df):
    """
    Displays a concise summary of a DataFrame, including dtypes and non-null values.

    Args:
        df (pandas.DataFrame): The DataFrame to inspect.
    """
    df.info()
