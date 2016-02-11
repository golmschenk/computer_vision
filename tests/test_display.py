from unittest.mock import patch
import numpy as np
import scipy

import constant
from display import draw_normal, draw_data_under_normal


class TestDisplay:
    @patch('display.plt')
    def test_draw_normal(self, mock_plt):
        mean = 1
        sigma = 1
        plotting_space = np.linspace(scipy.stats.norm.ppf(0.01, mean, sigma),
                                     scipy.stats.norm.ppf(0.99, mean, sigma),
                                     constant.plot_samples)
        distribution = scipy.stats.norm.pdf(plotting_space, mean, sigma)

        draw_normal(mean, sigma)

        assert np.array_equal((plotting_space, distribution), mock_plt.plot.call_args[0])

    @patch('display.plt')
    @patch('display.np.linspace')
    @patch('display.norm')
    def test_draw_normal_can_be_passed_a_color_for_the_normal(self, mock_norm, mock_linspace, mock_plt):
        draw_normal(0, 0, color='r')

        assert mock_plt.plot.call_args[1]['color'] == 'r'

    @patch('display.draw_normal')
    @patch('display.plt')
    def test_draw_data_under_normal(self, mock_plt, mock_draw_normal):
        data = [0.9, 1.0]
        mean = 1
        sigma = 1

        draw_data_under_normal(data, mean, sigma)

        assert (([0.9, 0.9], [0.0, scipy.stats.norm.pdf(0.9, mean, sigma)]),) in mock_plt.plot.call_args_list
        assert (([1.0, 1.0], [0.0, scipy.stats.norm.pdf(1.0, mean, sigma)]),) in mock_plt.plot.call_args_list
        assert mock_draw_normal.call_args == ((mean, sigma),)