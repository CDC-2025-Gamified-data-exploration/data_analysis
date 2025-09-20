# This file contains the function for plotting a correlation heatmap.

import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_heatmap(corr_matrix):
    """
    Plots a heatmap of a correlation matrix.

    Args:
        corr_matrix (pandas.DataFrame): A correlation matrix.
    """
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()
