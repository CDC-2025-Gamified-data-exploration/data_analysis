# This file contains the function for Random Forest-based imputation.

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

def impute_missforest(df):
    """
    Performs imputation using a Random Forest model, similar to R's missForest package.
    This uses scikit-learn's IterativeImputer with a RandomForest estimator.

    Args:
        df (pandas.DataFrame): The DataFrame with missing data.

    Returns:
        pandas.DataFrame: The DataFrame with imputed values.
    """
    imputer = IterativeImputer(estimator=RandomForestRegressor(), max_iter=10, random_state=0)
    imputed_data = imputer.fit_transform(df)
    return pd.DataFrame(imputed_data, columns=df.columns, index=df.index)
