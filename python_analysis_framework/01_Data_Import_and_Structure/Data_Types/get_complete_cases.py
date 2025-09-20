# This file contains the function for filtering for complete cases.

def get_complete_cases(df):
    """
    Returns the rows from the DataFrame that have no missing values.

    Args:
        df (pandas.DataFrame): The DataFrame to filter.

    Returns:
        pandas.DataFrame: A DataFrame with only complete rows.
    """
    return df.dropna()
