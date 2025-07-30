import numpy as np
import pandas as pd
import pyddm
from pyddm import gddm, Sample
from scipy.optimize import minimize

from fitting.costs_functions import minimizer_cost_function_rescorla_wagner, \
    minimizer_cost_function_reinforcement_learning, minimizer_cost_function_random_response, \
    minimizer_cost_function_win_stay_lose_switch
import logging
import pyddm.parameters as param

def minimizing_random_response(model_function, initial_parameters, rewards,
                               observed_data=None,
                               cost_metric='log-likelihood', minimize_options=None):
    """
    Function to minimize the cost_function using scipy.optimize.minimize.

    Parameters
    ----------
    model_function : function
        a model function of random response type.
    initial_parameters : list or tuple
        Initial guesses for the model parameters to be optimized.
    rewards : np.ndarray
        Array of rewards (ground truth).
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
    result = minimize(minimizer_cost_function_random_response,
                      args=(model_function, rewards, observed_data, cost_metric), x0=np.array(initial_parameters),
                      **minimize_options)

    return result


def minimizing_win_stay_lose_switch(model_function, initial_parameters, choices, rewards,
                                    observed_data=None,
                                    cost_metric='log-likelihood', minimize_options=None):
    """
    Function to minimize the cost function for the Win-Stay-Lose-Switch model using scipy.optimize.minimize.

    Parameters
    ----------
    model_function : function
        The Win-Stay-Lose-Switch model function.
    initial_parameters : list or tuple
        Initial guess for the epsilon parameter.
    choices : np.ndarray
        Binary array of choices (0 or 1).
    rewards : np.ndarray
        Binary array of rewards (0 or 1).
    observed_data : np.ndarray, optional
        Observed data (computed from choices if None).
    cost_metric : str
        The cost metric to use ('log-likelihood', 'mse', 'rmse', etc.)
    minimize_options : dict, optional
        Dictionary containing optimization settings.

    Returns
    -------
    result : OptimizeResult
        The result of the optimization.
    """
    if minimize_options is None:
        minimize_options = {
            'method': 'L-BFGS-B',
            'bounds': [(0.0, 1.0)],  # Epsilon must be between 0 and 1
            'options': {'disp': False}
        }

    result = minimize(minimizer_cost_function_win_stay_lose_switch,
                      args=(model_function, choices, rewards, observed_data, cost_metric),
                      x0=np.array(initial_parameters),
                      **minimize_options)

    return result


def minimizing_reinforcement_learning_model(model_function, initial_parameters, rewards, choices,
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
    result = minimize(minimizer_cost_function_reinforcement_learning,
                      args=(
                          model_function, rewards, choices, observed_data, cost_metric),
                      x0=np.array(initial_parameters), **minimize_options)

    return result


def minimizing_rescorla_wagner_model(model_function, initial_parameters, rewards, stimuli_present=None,
                                     extra_function_params=None,
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
    rewards : np.ndarray or None
        Array of rewards (ground truth).
    stimuli_present : np.ndarray
        Binary array indicating stimulus presence on each trial, if None assume that all the rewards had stimuli.
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

    result = minimize(minimizer_cost_function_rescorla_wagner,
                      args=(
                          model_function, rewards, stimuli_present, extra_function_params, observed_data, cost_metric),
                      x0=np.array(initial_parameters), **minimize_options)

    return result


def fit_ddm(rt_data, response_data, pyddm_options=None,longest_rt_duration=3,pyddm_cpu_n=1):
    """
    Fit a Drift Diffusion Model (DDM) to reaction time and choice data using PyDDM.

    Parameters
    ----------
    rt_data : np.ndarray
        Array of reaction times, the data should be in seconds, meaning 500 milliseconds is 0.5.
    response_data : np.ndarray
        Binary array of responses (0 or 1).
    pyddm_options : dict, optional
        Dictionary of parameter ranges for the model. If None, default ranges are used.

    Returns
    -------
    model : pyddm.Model
        The fitted PyDDM model.
    """
    # Create a pandas DataFrame with the data
    logging.getLogger('pyddm').setLevel(logging.ERROR)
    param.verbose = False
    param.renorm_warnings = False
    df = pd.DataFrame({
        'rt': rt_data,
        'correct': response_data
    })

    # Create a PyDDM sample from the DataFrame
    sample = Sample.from_pandas_dataframe(df, rt_column_name="rt", choice_column_name="correct")
    if pyddm_options is None:
        pyddm_options = {
            "driftrate": (-5, 5),
            "B": (0.1, 5),
            "x0": (-.5, .5),
            "ndt": (0.1, .3)
        }
    model = gddm(drift="driftrate", noise=1, bound="B", starting_position="x0", nondecision="ndt",
                 parameters=pyddm_options,dt=0.005,  # Default is 0.01, try 0.005 or 0.001
    dx=0.005,T_dur=longest_rt_duration)
    pyddm.set_N_cpus(pyddm_cpu_n)
    model.fit(sample,verbose=False)
    # Set default parameter ranges if not provided
    return model
    # parameters = []
    # for param in model.get_model_parameters():
    #     parameters.append(float(param))
    # return parameters


if __name__ == "__main__":
    # Example 1: Rescorla-Wagner model
    from computational_models.rescorla_wagner_simple import rescorla_wagner

    initial_parameters = [0.3, 0.1]
    rewards = np.array([1, 1, 0, 1, 0, 1, 1, 0, 0, 1])
    # Minimize options for scipy
    minimize_options = {
        'method': 'L-BFGS-B',  # Optimization method
        'tol': 1e-6,  # Convergence tolerance
        'options': {'disp': False}  # Display optimization progress
    }

    # Perform minimization
    result = minimizing_rescorla_wagner_model(
        model_function=rescorla_wagner,
        initial_parameters=initial_parameters,
        rewards=rewards,
        stimuli_present=None,
        cost_metric='log-likelihood',
        minimize_options=minimize_options
    )

    print("Example 1: Rescorla-Wagner Model")
    print(f"Optimized parameters: {result.x}")
    print(f"Final cost: {result.fun}")

    # Example 2: Drift Diffusion Model (DDM) using PyDDM

    # Example reaction times and responses for 10 trials
    rt_data = np.array([0.8, 1.2, 0.7, 1.5, 0.9, 1.1, 0.6, 1.3, 1.0, 0.5])
    response_data = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1])

    # Define parameter ranges for fitting
    parameter_ranges = {
        "driftrate": (-10, 10),
        "B": (0.1, 5),
        "x0": (-.5, .5),
        "ndt": (0, .5)
    }

    # Fit the model using PyDDM
    print("\nExample 2: Drift Diffusion Model (DDM) using PyDDM")
    print("Fitting model...")

    fitted_model = fit_ddm(rt_data, response_data, parameter_ranges)

    # Print the fitted parameters
    print("Fitted parameters:")
    for param_name, param_value in fitted_model.parameters().items():
        print(f"{param_name}: {param_value}")
