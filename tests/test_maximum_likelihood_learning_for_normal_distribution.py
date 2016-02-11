from unittest.mock import patch
import numpy as np
import scipy

import constant
from maximum_likelihood_learning_for_normal_distribution import generate_parameters, display_model


class TestMaximumLikelihoodLearningForNormalDistribution:
    def test_parameters_are_correctly_generated(self):
        training_data = [0.5, 0.7, 0.8, 1.0, 1.2, 1.3]

        mean, variance = generate_parameters(training_data)

        assert abs(mean - 0.9166) < 0.0001
        assert abs(variance - 0.0780) < 0.0001

    @patch('maximum_likelihood_learning_for_normal_distribution.display')
    @patch('maximum_likelihood_learning_for_normal_distribution.generate_parameters')
    @patch('maximum_likelihood_learning_for_normal_distribution.plt')
    def test_the_model_can_be_displayed(self, mock_plt, mock_generate_parameters, mock_display):
        mean = 0.9166
        sigma = 0.0780 ** 0.5
        mock_generate_parameters.return_value = (0.9166, 0.0780)
        training_data = [0.5, 0.7, 0.8, 1.0, 1.2, 1.3]

        display_model(training_data)

        assert (([0.7, 0.7], [0.0, scipy.stats.norm.pdf(0.7, mean, sigma)]),) in mock_plt.plot.call_args_list
        assert (([1.2, 1.2], [0.0, scipy.stats.norm.pdf(1.2, mean, sigma)]),) in mock_plt.plot.call_args_list
        assert mock_display.draw_normal.call_args == ((mean, sigma),)
        assert mock_plt.show.called
