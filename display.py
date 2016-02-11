"""
Code to help display the distributions and data.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import constant


def draw_normal(mean, sigma, color=None):
    """
    A function to consistently draw normal distributions the same way.

    :param mean: The mean of the normal distribution.
    :type mean: float
    :param sigma: The sigma of the normal distribution.
    :type sigma: float
    :param color: The color to display the distribution as.
    :type color: str
    """
    plotting_space = np.linspace(norm.ppf(0.01, mean, sigma),
                                 norm.ppf(0.99, mean, sigma),
                                 constant.plot_samples)
    distribution = norm.pdf(plotting_space, mean, sigma)
    if color:
        plt.plot(plotting_space, distribution, color=color)
    else:
        plt.plot(plotting_space, distribution)


def draw_data_under_normal(data, mean, sigma, data_color=None, normal_color=None):
    """
    Draws a normal along with data points underneath the normal.

    :param data: The data points to be drawn.
    :type data: list[float]
    :param mean: The mean of the normal distribution.
    :type mean: float
    :param sigma: The sigma of the normal distribution.
    :type sigma: float
    :return:
    :rtype:
    """
    for datum in data:
        if data_color:
            plt.plot([datum, datum], [0.0, norm.pdf(datum, mean, sigma)], color=data_color)
        else:
            plt.plot([datum, datum], [0.0, norm.pdf(datum, mean, sigma)])
    draw_normal(mean, sigma, color=normal_color)
