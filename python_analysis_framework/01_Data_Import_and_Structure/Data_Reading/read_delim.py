# This file contains the function for reading delimited files.

import pandas as pd

def read_delim(filepath, sep='\\t', **kwargs):
    """
    Reads a delimited file into a pandas DataFrame. Defaults to tab-separated.

    Args:
        filepath (str): The path to the file.
        sep (str): The delimiter to use.
        **kwargs: Additional keyword arguments to pass to pandas.read_csv().

    Returns:
        pandas.DataFrame: The loaded DataFrame.
    """
    return pd.read_csv(filepath, sep=sep, **kwargs)
