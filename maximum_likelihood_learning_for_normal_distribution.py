"""
Code for the maximum likelihood learning for a normal distribution.
"""


def maximum_likelihood_learning_for_normal_distribution(training_data):
    """
    Determines the maximum likelihood parameters (mean and variance) for a normal distribution fit to a set of
    training data.

    :param training_data: The training data to fit the distribution to.
    :type training_data: list[float]
    :return: The mean and variance of the normal distribution.
    :rtype: float, float
    """
    mean = sum(training_data)/len(training_data)
    variance = sum([(training_datum - mean) ** 2 for training_datum in training_data])/len(training_data)
    return mean, variance
