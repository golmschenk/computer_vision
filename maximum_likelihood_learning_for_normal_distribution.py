def maximum_likelihood_learning_for_normal_distribution(training_data):
    mean = sum(training_data)/len(training_data)
    variance = sum([(training_datum - mean) ** 2 for training_datum in training_data])/len(training_data)
    return mean, variance
