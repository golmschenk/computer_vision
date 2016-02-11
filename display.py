"""
Code to help display the distributions and data.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import constant


def draw_normal(mean, sigma):
    """
    A function to consistently draw normal distributions the same way.

    :param mean: The mean of the normal distribution.
    :type mean: float
    :param sigma: The sigma of the normal distribution.
    :type sigma: float
    """
    plotting_space = np.linspace(norm.ppf(0.01, mean, sigma),
                                 norm.ppf(0.99, mean, sigma),
                                 constant.plot_samples)
    distribution = norm.pdf(plotting_space, mean, sigma)
    plt.plot(plotting_space, distribution)