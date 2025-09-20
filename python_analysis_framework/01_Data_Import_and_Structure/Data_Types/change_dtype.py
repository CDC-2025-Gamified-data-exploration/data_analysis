# This file contains the function for changing the data type of a column.

def change_dtype(series, dtype):
    """
    Casts a pandas object to a specified dtype.

    Args:
        series (pandas.Series): The Series whose data type will be changed.
        dtype (str or numpy.dtype): The target data type.

    Returns:
        pandas.Series: The Series with the new data type.
    """
    return series.astype(dtype)
