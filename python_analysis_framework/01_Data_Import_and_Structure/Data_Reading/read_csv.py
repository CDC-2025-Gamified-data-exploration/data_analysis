# This file contains the function for reading CSV files.

import pandas as pd

def read_csv(filepath, **kwargs):
    """
    Reads a comma-separated values (csv) file into a pandas DataFrame.

    Args:
        filepath (str): The path to the CSV file.
        **kwargs: Additional keyword arguments to pass to pandas.read_csv().

    Returns:
        pandas.DataFrame: The loaded DataFrame.
    """
    return pd.read_csv(filepath, **kwargs)
