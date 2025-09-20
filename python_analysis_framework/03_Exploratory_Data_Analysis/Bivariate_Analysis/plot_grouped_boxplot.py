# This file contains the function for plotting grouped boxplots.

import matplotlib.pyplot as plt
import seaborn as sns

def plot_grouped_boxplot(df, cat_col, cont_col):
    """
    Plots boxplots of a continuous variable grouped by a categorical variable.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        cat_col (str): The name of the categorical column for grouping.
        cont_col (str): The name of the continuous column for the values.
    """
    sns.boxplot(data=df, x=cat_col, y=cont_col)
    plt.title(f'Box Plot of {cont_col} by {cat_col}')
    plt.xlabel(cat_col)
    plt.ylabel(cont_col)
    plt.show()
