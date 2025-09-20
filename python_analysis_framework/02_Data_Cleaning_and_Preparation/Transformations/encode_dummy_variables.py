# This file contains the function for creating dummy (one-hot encoded) variables.

import pandas as pd

def encode_dummy_variables(df, columns):
    """
    Converts categorical variables into dummy/indicator variables.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        columns (list of str): A list of column names to encode.

    Returns:
        pandas.DataFrame: The DataFrame with encoded variables.
    """
    return pd.get_dummies(df, columns=columns, drop_first=True)
