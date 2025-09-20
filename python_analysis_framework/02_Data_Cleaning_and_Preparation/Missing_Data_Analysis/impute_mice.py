# This file contains the function for MICE imputation.

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd

def impute_mice(df):
    """
    Performs Multiple Imputation by Chained Equations (MICE).

    Args:
        df (pandas.DataFrame): The DataFrame with missing data.

    Returns:
        pandas.DataFrame: The DataFrame with imputed values.
    """
    imputer = IterativeImputer(max_iter=10, random_state=0)
    imputed_data = imputer.fit_transform(df)
    return pd.DataFrame(imputed_data, columns=df.columns, index=df.index)
