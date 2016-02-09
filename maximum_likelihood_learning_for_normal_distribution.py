"""
Code for the maximum likelihood learning for a normal distribution.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import seaborn as sns

import constant


def generate_parameters(training_data):
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


def display_model(training_data):
    """
    Displays a plot of the maximum likelihood distribution along with the input data.

    :param training_data: The training data to fit the distribution to.
    :type training_data: list[float]
    """
    mean, variance = generate_parameters(training_data)
    sigma = variance ** 0.5
    for training_datum in training_data:
        plt.plot([training_datum, training_datum], [0.0, norm.pdf(training_datum, mean, sigma)])
    plotting_space = np.linspace(norm.ppf(0.01, mean, sigma),
                                 norm.ppf(0.99, mean, sigma),
                                 constant.plot_samples)
    plt.plot(plotting_space, norm.pdf(plotting_space, mean, sigma))
    plt.show()


if __name__ == '__main__':
    display_model([0.5, 0.7, 0.8, 1.0, 1.2, 1.3])
