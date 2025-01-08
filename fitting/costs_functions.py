import numpy as np


def cost_function(model_function, parameters, rewards, stimuli_present, observed_data=None, cost_metric='log-likelihood'):
    """
    General cost function for comparing predicted probabilities to observed data.
    """
    if observed_data is None:
        observed_data = np.array(rewards)
    V_history = model_function(parameters, rewards, stimuli_present)
    V_present = np.sum(V_history[0:] * stimuli_present, axis=1)
    # Convert V_history to choice probabilities using logistic (sigmoid) function
    p_choice = sigmoid

    # Calculate cost based on the specified cost metric
    if cost_metric == 'log-likelihood':
        return log_likelihood(observed_data, p_choice)
    elif cost_metric == 'mse':
        return mse(observed_data, p_choice)
    else:
        raise ValueError(f"Unsupported cost metric: {cost_metric}")
