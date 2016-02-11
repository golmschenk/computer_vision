"""
Code for the maximum a posterior learning for normal parameters.
"""
import matplotlib.pyplot as plt
import seaborn as sns

import display
from maximum_likelihood_learning_for_normal_distribution import \
    generate_parameters as generate_maximum_likelihood_parameters


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


def display_model(training_data, alpha, beta, gamma, delta):
    """
    Displays a plot of the data under the maximum a posterior distribution, along side the maximum likelihood for
    the data and the prior.

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
    """
    prior_mean, prior_variance = generate_parameters([], alpha, beta, gamma, delta)
    ml_mean, ml_variance = generate_maximum_likelihood_parameters(training_data)
    mean, variance = generate_parameters(training_data, alpha, beta, gamma, delta)

    prior_sigma = prior_variance ** 0.5
    ml_sigma = ml_variance ** 0.5
    sigma = variance ** 0.5

    map_color = sns.color_palette()[0]
    data_color = sns.color_palette()[1]
    prior_color = sns.color_palette()[2]
    ml_color = sns.color_palette()[3]

    display.draw_normal(prior_mean, prior_sigma, color=prior_color)
    display.draw_normal(ml_mean, ml_sigma, color=ml_color)
    display.draw_data_under_normal(training_data, mean, sigma, data_color=data_color, normal_color=map_color)
    plt.show()


if __name__ == '__main__':
    display_model([0.5, 0.7, 0.8, 1.0, 1.2, 1.3], 1, 1, 1, 0)
