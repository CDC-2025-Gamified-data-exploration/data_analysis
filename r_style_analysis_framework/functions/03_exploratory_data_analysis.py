# Category 3: Exploratory Data Analysis
# This module contains functions for exploring data through visualization and summary statistics.

# --- Univariate Analysis ---

def plot_histogram(series, bins='auto'):
    """
    Equivalent to R's hist()
    Uses matplotlib.pyplot.hist() or seaborn.histplot()
    - Libraries: matplotlib, seaborn
    """
    pass

def plot_density(series):
    """
    Equivalent to R's density()
    Uses seaborn.kdeplot()
    - Libraries: seaborn
    """
    pass

def plot_qq(series, dist='norm'):
    """
    Equivalent to R's qqnorm()
    Uses statsmodels.api.qqplot()
    - Libraries: statsmodels
    """
    pass

def get_summary_statistics(series):
    """
    Equivalent to R's summary() on a vector, plus skew/kurtosis
    - Calculates mean, median, sd, var, range, IQR, quantiles, skew, kurtosis
    - Libraries: pandas, scipy.stats
    """
    pass

# --- Bivariate Analysis ---

def plot_scatter(df, x_col, y_col):
    """
    Equivalent to R's plot(y ~ x)
    Uses seaborn.scatterplot()
    - Libraries: seaborn
    """
    pass

def calculate_correlation(series1, series2, method='pearson'):
    """
    Equivalent to R's cor()
    - Supports 'pearson', 'spearman', 'kendall'
    - Libraries: scipy.stats
    """
    pass

def plot_grouped_boxplot(df, cat_col, cont_col):
    """
    Equivalent to R's boxplot(y ~ x)
    Uses seaborn.boxplot()
    - Libraries: seaborn
    """
    pass

def test_chi_square_contingency(df, cat_col1, cat_col2):
    """
    Equivalent to R's chisq.test() on a table
    Uses scipy.stats.chi2_contingency()
    - Libraries: scipy.stats, pandas
    """
    pass

# --- Multivariate Analysis ---

def plot_correlation_matrix(df):
    """
    Equivalent to R's corrplot::corrplot()
    Uses seaborn.heatmap() on a df.corr() matrix
    - Libraries: seaborn, pandas
    """
    pass

def plot_pairs_matrix(df, columns_to_plot, hue_col=None):
    """
    Equivalent to R's GGally::ggpairs() or pairs()
    Uses seaborn.pairplot()
    - Libraries: seaborn
    """
    pass

def plot_3d_scatter(df, x_col, y_col, z_col, color_col=None):
    """
    Equivalent to R's plotly::plot_ly() for 3D scatter
    Uses plotly.express.scatter_3d()
    - Libraries: plotly
    """
    pass
