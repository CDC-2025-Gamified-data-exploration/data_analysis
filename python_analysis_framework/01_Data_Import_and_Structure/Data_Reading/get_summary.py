# This file contains the function for generating a descriptive summary of a DataFrame.

def get_summary(df):
    """
    Generates descriptive statistics for a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to summarize.

    Returns:
        pandas.DataFrame: A summary DataFrame.
    """
    return df.describe(include='all')
