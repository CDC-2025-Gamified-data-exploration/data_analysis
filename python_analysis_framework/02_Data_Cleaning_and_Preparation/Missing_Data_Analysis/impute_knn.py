# This file contains the function for k-NN imputation.

from sklearn.impute import KNNImputer
import pandas as pd

def impute_knn(df, n_neighbors=5):
    """
    Performs k-Nearest Neighbors imputation.

    Args:
        df (pandas.DataFrame): The DataFrame with missing data.
        n_neighbors (int): Number of neighboring samples to use for imputation.

    Returns:
        pandas.DataFrame: The DataFrame with imputed values.
    """
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_data = imputer.fit_transform(df)
    return pd.DataFrame(imputed_data, columns=df.columns, index=df.index)
