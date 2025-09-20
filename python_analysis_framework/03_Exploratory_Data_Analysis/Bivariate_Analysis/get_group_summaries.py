# This file contains the function for getting summary statistics by group.

def get_group_summaries(df, group_col, value_col):
    """
    Calculates summary statistics for a numerical column, grouped by a categorical column.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        group_col (str): The name of the categorical column to group by.
        value_col (str): The name of the numerical column to summarize.

    Returns:
        pandas.DataFrame: A DataFrame with summary statistics for each group.
    """
    return df.groupby(group_col)[value_col].describe()
