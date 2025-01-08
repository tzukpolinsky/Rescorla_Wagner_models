import numpy as np
from scipy.optimize import minimize

def fit_parameters(model_function,
                   rewards: np.ndarray,
                   stimuli_present: np.ndarray,
                   cost_metric: str = 'log-likelihood',
                   observed_data: np.ndarray = None,
                   minimizer_params_dict: dict = None):
    """
    Minimizes the chosen cost_metric by varying alpha, beta.

    Parameters
    ----------
    model_function : callable
        E.g., rescorla_wagner. Accepts (alpha, beta, rewards, stimuli_present).
    rewards : np.ndarray
        Rewards array, shape=(n_trials,).
    stimuli_present : np.ndarray
        Binary array, shape=(n_trials, n_stimuli).
    cost_metric : {'log-likelihood', 'MSE'}
        Which cost metric to minimize.
    observed_data : np.ndarray or None
        If None, synthetic data is generated internally (purely for demonstration).
        Otherwise, must have shape=(n_trials,).
    minimizer_params_dict : dict, optional
        Additional parameters to pass to scipy.optimize.minimize (e.g. method, options).

    Returns
    -------
    result : OptimizeResult
        The object returned by scipy.optimize.minimize, containing best-fit params, etc.
    """
    if minimizer_params_dict is None:
        minimizer_params_dict = {}

    # Initial guess for alpha, beta
    initial_guess = [0.1, 1.0]

    # Bounds: alpha in [0,1], beta >= 0
    bounds = [(0.0, 1.0), (0.0, None)]

    result = minimize(
        fun=,
        x0=initial_guess,
        args=(model_function, rewards, stimuli_present, cost_metric, observed_data),
        bounds=bounds,
        **minimizer_params_dict
    )

    return result
