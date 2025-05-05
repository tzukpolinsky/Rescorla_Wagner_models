import numpy as np


def random_response(b: float, data: np.ndarray) -> np.ndarray:
    """
    Random Responding Model for existing data.

    Parameters
    ----------
    b : float
        Bias parameter. Probability of responding "1".
    data : np.ndarray
        Binary array of shape (n_trials,) representing observed responses (0 or 1).

    Returns
    -------
    probabilities : np.ndarray of shape (n_trials,)
        Probability of observed responses given the model.

    Notes
    -----
    This function evaluates the random responding model on existing data.
    The returned probabilities can be used to compute log-likelihood or RMSE.
    """
    if not (0 <= b <= 1):
        raise ValueError("Bias parameter b must be between 0 and 1.")

    if not np.all(np.isin(data, [0, 1])):
        raise ValueError("Data must be a binary array containing only 0 and 1.")

    # Calculate probabilities for the observed responses
    probabilities = np.where(data == 1, b, 1 - b)

    return probabilities
