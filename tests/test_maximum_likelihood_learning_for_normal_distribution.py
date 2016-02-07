from maximum_likelihood_learning_for_normal_distribution import maximum_likelihood_learning_for_normal_distribution


def test_maximum_likelihood_learning_for_normal_distribution():
    training_data = [0.5, 0.7, 0.8, 1.0, 1.2, 1.3]

    mean, variance = maximum_likelihood_learning_for_normal_distribution(training_data)

    assert abs(mean - 0.9166) < 0.0001
    assert abs(variance - 0.0780) < 0.0001
