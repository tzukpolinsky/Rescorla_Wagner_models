import numpy as np


def log_likelihood(observed_data: np.array, predicted_data: np.array, eps=1e-9):
    """
    Calculate the log-likelihood cost metric.
    """
    ll_array = observed_data * np.log(predicted_data + eps) \
               + (1 - observed_data) * np.log(1 - predicted_data + eps)
    return -np.sum(ll_array)
