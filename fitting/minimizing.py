import numpy as np
from scipy.optimize import minimize

from fitting.costs_functions import minimizer_cost_function


def minimizing_function(model_function, initial_parameters, rewards, stimuli_present, extra_function_params=None,
                        observed_data=None,
                        cost_metric='log-likelihood', minimize_options=None):
    """
    Function to minimize the cost_function using scipy.optimize.minimize.

    Parameters
    ----------
    model_function : function
        The model function generating associative strengths.
    initial_parameters : list or tuple
        Initial guesses for the model parameters to be optimized.
    rewards : np.ndarray
        Array of rewards (ground truth).
    stimuli_present : np.ndarray
        Binary array indicating stimulus presence on each trial.
    observed_data : np.ndarray, optional
        Observed data (computed from rewards if None).
    cost_metric : str
        The cost metric to use ('log-likelihood', 'mse', 'rmse', 'meanabs', 'medianabs', 'maxabs').
    minimize_options : dict, optional
        Dictionary containing optimization settings (e.g., method, tolerance, display options, etc.).

    Returns
    -------
    result : OptimizeResult
        The result of the optimization.
    """
    if minimize_options is None:
        minimize_options = {}

    result = minimize(minimizer_cost_function,
                      args=(
                      model_function, rewards, stimuli_present, extra_function_params, observed_data, cost_metric),
                      x0=np.array(initial_parameters), **minimize_options)

    return result


if __name__ == "__main__":
    from models.rescorla_wagner_simple import rescorla_wagner

    initial_parameters = [0.9, 0.5]
    rewards = np.array([1, 1, 0, 1, 0, 1, 1, 0, 0, 1])

    # Minimize options for scipy
    minimize_options = {
        'method': 'L-BFGS-B',  # Optimization method
        'tol': 1e-6,  # Convergence tolerance
        'options': {'disp': False}  # Display optimization progress
    }

    # Perform minimization
    result = minimizing_function(
        model_function=rescorla_wagner,
        initial_parameters=initial_parameters,
        rewards=rewards,
        stimuli_present=None,
        cost_metric='log-likelihood',
        minimize_options=minimize_options
    )

    print(f"Optimized parameters: {result.x}")
    print(f"Final cost: {result.fun}")
