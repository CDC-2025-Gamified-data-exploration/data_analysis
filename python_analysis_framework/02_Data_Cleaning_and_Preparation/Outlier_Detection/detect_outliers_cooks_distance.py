# This file contains the function for calculating Cook's distance to find influential points.

def get_cooks_distance(model_results):
    """
    Calculates Cook's distance from a fitted statsmodels regression model.
    While this is a regression diagnostic, it's used to identify influential outliers.

    Args:
        model_results: A fitted model object from statsmodels.

    Returns:
        numpy.ndarray: An array of Cook's distances.
    """
    influence = model_results.get_influence()
    return influence.cooks_distance[0]
