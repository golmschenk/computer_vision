"""
Code for the maximum a posterior learning for normal parameters.
"""


def generate_parameters(training_data, alpha, beta, gamma, delta):
    """
    Determines the maximum a posterior parameters (mean and variance) for a normal distribution fit to a set of
    training data and a prior.

    :param training_data: The training data to fit the distribution to.
    :type training_data: list[float]
    :param alpha: The alpha hyperparameter of the prior
    :type alpha: float
    :param beta: The beta hyperparameter of the prior
    :type beta: float
    :param gamma: The gamma hyperparameter of the prior
    :type gamma: float
    :param delta: The delta hyperparameter of the prior
    :type delta: float
    :return: The mean and variance of the normal distribution.
    :rtype: float, float
    """
    mean = (sum(training_data) + gamma * delta) / (len(training_data) + gamma)
    variance = (sum([(training_datum - mean) ** 2 for training_datum in training_data]) +
                2 * beta + gamma * (delta - mean) ** 2) / (len(training_data) + 3 + 2 * alpha)
    return mean, variance
