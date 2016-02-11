from unittest.mock import patch
import numpy as np
import scipy

import constant
from display import draw_normal


@patch('display.plt')
class TestDisplay:
    def test_draw_normal(self, mock_plt):
        mean = 1
        sigma = 1
        plotting_space = np.linspace(scipy.stats.norm.ppf(0.01, mean, sigma),
                                     scipy.stats.norm.ppf(0.99, mean, sigma),
                                     constant.plot_samples)
        distribution = scipy.stats.norm.pdf(plotting_space, mean, sigma)

        draw_normal(mean, sigma)

        assert np.array_equal((plotting_space, distribution), mock_plt.plot.call_args_list[0][0])