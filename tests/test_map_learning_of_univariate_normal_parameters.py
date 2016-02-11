from unittest.mock import patch

from map_learning_of_univariate_normal_parameters import generate_parameters, display_model


class TestMapLearningOfUnivariateNormalParameters:
    def test_parameters_are_correctly_generated(self):
        training_data = [0.5, 0.7]
        alpha = 1
        beta = 1
        gamma = 1
        delta = 0

        mean, variance = generate_parameters(training_data, alpha, beta, gamma, delta)

        assert abs(mean - 0.4) < 0.0001
        assert abs(variance - 0.3229) < 0.0001

    @patch('map_learning_of_univariate_normal_parameters.display')
    @patch('map_learning_of_univariate_normal_parameters.generate_parameters')
    @patch('map_learning_of_univariate_normal_parameters.generate_maximum_likelihood_parameters')
    @patch('map_learning_of_univariate_normal_parameters.plt')
    def test_the_model_can_be_displayed(self, mock_plt, mock_ml_parameters, mock_generate_parameters, mock_display):
        mean = 0.8
        variance = 0.08
        mock_generate_parameters.side_effect = [(0, 1), (mean, variance)]
        mock_ml_parameters.return_value = (1, 0.07)
        training_data = [0.5, 0.7, 0.8, 1.0, 1.2, 1.3]

        display_model(training_data, 1, 2, 3, 4)

        assert mock_generate_parameters.call_args_list[0][0] == ([], 1, 2, 3, 4)
        assert mock_display.draw_normal.call_args_list[0][0] == (0, 1 ** 0.5)
        assert mock_display.draw_normal.call_args_list[1][0] == (1, 0.07 ** 0.5)
        assert mock_display.draw_data_under_normal.call_args[0] == (training_data, mean, variance ** 0.5)
        assert mock_plt.show.called
