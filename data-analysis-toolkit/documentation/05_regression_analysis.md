# Tutorial: Regression Analysis & Diagnostics

This tutorial walks through the process of building, diagnosing, and validating a regression model using the toolkit.

## 1. Fitting a Model

The first step is to fit a model. The `functions/linear_regression.py` module contains functions for this.

```python
from functions import linear_regression
import pandas as pd

# Assume df is your cleaned DataFrame
# predictors = ['age', 'experience', 'education_level']
# target = 'salary'

# Fit a multiple linear regression model
# model_results = linear_regression.fit_multiple_linear_regression(df, x_cols=predictors, y_col=target)
# print(model_results.summary())
```

## 2. Diagnosing the Model

Model diagnostics are crucial for assessing the validity of your model's results. The `functions/residual_analysis.py` and `functions/influence_diagnostics.py` modules are key here.

### Standard Diagnostic Plots
A great place to start is the standard dashboard of four diagnostic plots.
```python
from functions import plotting

# Generate the diagnostic dashboard
# fig = plotting.create_regression_diagnostic_dashboard(model_results)
# fig.tight_layout()
# fig.show()
```

### Checking for Influential Points
You can check for influential points using Cook's distance.
```python
from functions import influence_diagnostics

# Get the influence object
# influence = influence_diagnostics.get_influence_summary(model_results)

# Plot Cook's distance
# influence_diagnostics.plot_cooks_distance(influence)
```

## 3. Testing Assumptions

You should also formally test the key assumptions of linear regression. Functions are in `functions/assumption_testing.py`.

```python
from functions import assumption_testing

# Test for heteroscedasticity using the Breusch-Pagan test
# lm_stat, p_value, _, _ = assumption_testing.test_breusch_pagan(model_results)
# if p_value < 0.05:
#     print("Evidence of heteroscedasticity was found.")
```

## 4. Validating the Model

Finally, validate your model's predictive performance using cross-validation. Functions are in `functions/cross_validation.py`.

```python
from functions import cross_validation
from sklearn.linear_model import LinearRegression

# X = df[predictors]
# y = df[target]
# model = LinearRegression()

# Perform 10-fold cross-validation
# scores = cross_validation.perform_k_fold_cv(model, X, y, k=10)
# print(f"Mean CV R-squared: {scores.mean():.2f}")
```
