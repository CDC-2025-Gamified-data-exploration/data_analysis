# Tutorial: Statistical Testing

This tutorial explains how to use the statistical testing suite in the toolkit. These functions allow you to perform formal hypothesis tests.

## 1. Normality Tests

Before running many parametric tests, you should check if your data is normally distributed. Functions are located in `functions/normality_tests.py`.

```python
from functions import normality_tests
import pandas as pd

# Assume 'data_series' is a pandas Series
# stat, p_value = normality_tests.test_shapiro_wilk(data_series)
# if p_value > 0.05:
#     print("Data looks normally distributed.")
# else:
#     print("Data does not look normally distributed.")
```

## 2. Comparing Two Samples

To compare the means of two groups, you can use a t-test (if assumptions are met) or a non-parametric alternative. Functions are in `functions/two_sample_tests.py`.

```python
from functions import two_sample_tests

# sample_a = df[df['group'] == 'A']['value']
# sample_b = df[df['group'] == 'B']['value']

# Perform an independent t-test
# t_stat, p_value = two_sample_tests.test_independent_t(sample_a, sample_b)
# print(f"T-test p-value: {p_value}")

# Perform a Mann-Whitney U test
# u_stat, p_value = two_sample_tests.test_mann_whitney_u(sample_a, sample_b)
# print(f"Mann-Whitney U test p-value: {p_value}")
```

## 3. Comparing Multiple Samples

When you have more than two groups, you can use ANOVA or the Kruskal-Wallis test. Functions are in `functions/multiple_sample_tests.py`.

```python
from functions import multiple_sample_tests

# sample_a = df[df['group'] == 'A']['value']
# sample_b = df[df['group'] == 'B']['value']
# sample_c = df[df['group'] == 'C']['value']

# Perform a one-way ANOVA
# f_stat, p_value = multiple_sample_tests.test_one_way_anova(sample_a, sample_b, sample_c)
# if p_value < 0.05:
#     print("There is a significant difference between at least two groups.")
#     # You would then run a post-hoc test to see which groups are different.
#     # from functions import run_posthoc_tukey
#     # posthoc_results = run_posthoc_tukey(df, group_col='group', value_col='value')
#     # print(posthoc_results)
```
