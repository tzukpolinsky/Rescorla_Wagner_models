import numpy as np
from statsmodels.tools.eval_measures import mse, rmse, meanabs, medianabs, maxabs

from Rescorla_models import random_response
from math_functions.conversion_functions import sigmoid
from math_functions.costs_metrics import log_likelihood


def metric_calculation(p_choice, observed_data, cost_metric='log-likelihood'):
    if cost_metric == 'log-likelihood':
        return log_likelihood(observed_data, p_choice)
    elif cost_metric == 'mse':
        return mse(observed_data, p_choice)
    elif cost_metric == 'mse':
        return mse(observed_data, p_choice)
    elif cost_metric == 'rmse':
        return rmse(observed_data, p_choice)
    elif cost_metric == 'meanabs':
        return meanabs(observed_data, p_choice)
    elif cost_metric == 'medianabs':
        return medianabs(observed_data, p_choice)
    elif cost_metric == 'maxabs':
        return maxabs(observed_data, p_choice)
    else:
        raise ValueError(f"Unsupported cost metric: {cost_metric}")


def minimizer_cost_function_random_response(parameters, model_function, rewards,
                                            observed_data=None,
                                            cost_metric='log-likelihood'):
    """
    General cost function for comparing predicted probabilities to observed data.
    """

    p_choice = model_function(*parameters, rewards)
    if observed_data is None:
        observed_data = np.random.binomial(1, p_choice)
    return metric_calculation(p_choice, observed_data, cost_metric)


def minimizer_cost_function_rescorla_wagner(parameters, model_function, rewards, stimuli_present=None,
                                            extra_function_params=None,
                                            observed_data=None,
                                            cost_metric='log-likelihood'):
    """
    General cost function for comparing predicted probabilities to observed data.
    """
    n_trials = len(rewards)
    if observed_data is None:
        observed_data = np.array(rewards)
    if extra_function_params is not None:
        V_history,stimuli_present_output = model_function(*parameters, rewards, stimuli_present, *extra_function_params)
    else:
        V_history,stimuli_present_output = model_function(*parameters, rewards, stimuli_present)
    V_present = np.sum(V_history[1:] * stimuli_present_output, axis=1)
    p_choice = sigmoid(V_present)
    return metric_calculation(p_choice, observed_data, cost_metric=cost_metric)


def minimizer_cost_function_reinforcement_learning(parameters, model_function, rewards, choices, observed_data=None,
                                                   cost_metric='log-likelihood'):
    """
    General cost function for reinforcement learning models comparing predicted probabilities to observed data.
    Parameters:
    ----------
    - parameters: List of model parameters (e.g., [alpha, beta]).
    - model_function: Function that computes the predicted choice probabilities Pchoice.
    - rewards: List of rewards (0 or 1).
    - choices: List of chosen actions (0 or 1).
    - observed_data: Binary array of observed choices (if None, defaults to `choices`).
    - cost_metric: The cost metric to use ('log-likelihood', 'mse', 'rmse', 'meanabs', 'medianabs', 'maxabs').
    Returns:
    -------
    - Cost value according to the specified metric.
    """
    if observed_data is None:
        observed_data = np.array(choices)

    # Get predicted choice probabilities Pchoice from the model
    Pchoice = model_function(parameters, choices, rewards)

    # Ensure observed data is binary (0 or 1)
    observed_data = np.asarray(observed_data, dtype=int)
    # Calculate cost based on the specified cost metric
    return metric_calculation(Pchoice, observed_data, cost_metric=cost_metric)
