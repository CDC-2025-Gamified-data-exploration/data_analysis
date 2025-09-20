# This file contains the function for ordinal encoding.

from sklearn.preprocessing import OrdinalEncoder

def encode_ordinal(df, columns, categories='auto'):
    """
    Encodes categorical features as an integer array.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        columns (list of str): A list of column names to encode.
        categories (list of lists or 'auto'): The categories for each column.

    Returns:
        numpy.ndarray: The transformed data.
    """
    encoder = OrdinalEncoder(categories=categories)
    return encoder.fit_transform(df[columns])
