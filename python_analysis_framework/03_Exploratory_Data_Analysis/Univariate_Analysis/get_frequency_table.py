# This file contains the function for creating a frequency table.

def get_frequency_table(series, normalize=False):
    """
    Generates a frequency table (counts or proportions) for a categorical Series.

    Args:
        series (pandas.Series): The categorical data series.
        normalize (bool): If True, returns proportions instead of counts.

    Returns:
        pandas.Series: A Series containing the frequency table.
    """
    return series.value_counts(normalize=normalize)
