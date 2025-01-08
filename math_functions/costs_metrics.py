import numpy as np


def log_likelihood(observed_data: np.array, predicted_data: np.array, eps=1e-9):
    """
    Calculate the log-likelihood cost metric.
    """
    likelihoods = np.where(observed_data == 1, predicted_data + eps, 1 - predicted_data + eps)
    log_likelihoods = np.log(likelihoods)
    return -np.sum(log_likelihoods)


def mse(observed_data: np.array, predicted_data: np.array):
    """
    Calculate the Mean Squared Error (MSE).
    """
    return np.mean((observed_data - predicted_data) ** 2)
