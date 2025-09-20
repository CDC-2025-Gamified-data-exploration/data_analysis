# Tutorial: Data Preparation & Cleaning

This tutorial covers the use of functions for data preparation and cleaning. These functions are essential for preparing a dataset for analysis.

## 1. Missing Data Analysis

The `functions/missing_data.py` module provides tools for finding and handling missing values.

### Visualizing Missing Data
Before imputing, it's crucial to understand the pattern of missingness.
```python
from functions import missing_data
import pandas as pd

# Assume df is your DataFrame
# missing_data.plot_missing_data_matrix(df)
```

### Imputing Missing Data
The toolkit provides several imputation methods.
```python
# Impute with the mean
# df_imputed = missing_data.impute_mean(df, 'column_name')

# Impute with KNN
# df_imputed_knn = missing_data.impute_knn(df)
```

## 2. Outlier Detection

The `functions/outlier_detection.py` module helps identify and handle outliers.

### Detecting Outliers
You can use statistical methods like the IQR score to flag outliers.
```python
from functions import outlier_detection

# Detect outliers in 'column_name'
# outliers_df = outlier_detection.detect_outliers_iqr(df, 'column_name')
# print(outliers_df[outliers_df['is_outlier'] == True])
```

### Visualizing Outliers
A boxplot is a simple way to see outliers.
```python
# outlier_detection.plot_boxplot_outliers(df, 'column_name')
```

## 3. Data Transformation

The `functions/data_transformation.py` module contains functions for scaling and transforming variables.

### Scaling Numerical Data
Standardizing features to have a mean of 0 and a standard deviation of 1 is a common requirement for many models.
```python
from functions import data_transformation

# Scale multiple columns using Z-score standardization
# scaled_df = data_transformation.scale_z_score(df, columns=['col1', 'col2'])
```

### Encoding Categorical Data
Convert categorical variables into a numerical format.
```python
# Perform one-hot encoding on 'category_column'
# encoded_df = data_transformation.encode_one_hot(df, 'category_column')
```
