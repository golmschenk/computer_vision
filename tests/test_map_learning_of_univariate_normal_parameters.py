from map_learning_of_univariate_normal_parameters import generate_parameters


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
