# This file contains the function for reading Excel files.

import pandas as pd

def read_excel(filepath, **kwargs):
    """
    Reads an Excel file into a pandas DataFrame.

    Args:
        filepath (str): The path to the Excel file.
        **kwargs: Additional keyword arguments to pass to pandas.read_excel().

    Returns:
        pandas.DataFrame: The loaded DataFrame.
    """
    return pd.read_excel(filepath, **kwargs)
