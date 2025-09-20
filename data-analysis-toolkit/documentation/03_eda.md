# Tutorial: Exploratory Data Analysis (EDA)

This tutorial demonstrates how to use the toolkit's functions to perform Exploratory Data Analysis (EDA). EDA is a critical step for understanding the underlying structure of the data.

## 1. Univariate Analysis

Univariate analysis focuses on a single variable at a time. The relevant functions are in `functions/univariate_analysis.py`.

### Continuous Variables
For continuous variables, we are often interested in their distribution and central tendency.
```python
from functions import univariate_analysis
import pandas as pd

# Assume df is your DataFrame
# univariate_analysis.plot_histogram(df, 'age')
# univariate_analysis.plot_qq(df, 'age', dist='norm')
```

### Categorical Variables
For categorical variables, we typically look at frequencies.
```python
# univariate_analysis.plot_barchart(df, 'department')
```

## 2. Bivariate Analysis

Bivariate analysis explores the relationship between two variables. The relevant functions are in `functions/bivariate_analysis.py`.

### Continuous vs. Continuous
A scatter plot is the classic way to visualize this relationship.
```python
from functions import bivariate_analysis

# bivariate_analysis.plot_scatter(df, x_col='age', y_col='salary')
```

### Continuous vs. Categorical
Box plots are excellent for comparing distributions across groups.
```python
# bivariate_analysis.plot_grouped_boxplot(df, cat_col='department', cont_col='salary')
```

### Categorical vs. Categorical
A heatmap of a cross-tabulation can reveal associations.
```python
# bivariate_analysis.plot_crosstab_heatmap(df, 'department', 'region')
```

## 3. Multivariate Analysis

Multivariate analysis looks at relationships among three or more variables. The relevant functions are in `functions/multivariate_analysis.py`.

A pairs plot is a great way to get a quick overview of the relationships between all numerical variables.
```python
from functions import multivariate_analysis

# numerical_cols = ['age', 'salary', 'years_experience']
# multivariate_analysis.plot_pairs(df, columns=numerical_cols, hue_col='department')
```
