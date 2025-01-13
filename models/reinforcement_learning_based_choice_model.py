import numpy as np


def reinforcement_learning_simple_model(params, choices, rewards):
    """
    Calculate the negative log-likelihood for a simple reinforcement learning model.

    Parameters:
    - choices (list): List of choices (0 or 1) for each trial.
    - rewards (list): List of rewards (0 or 1) for each trial.
    - params (list): [alpha, beta] where
        alpha: learning rate
        beta: inverse temperature (decision precision)

    Returns:
    - float: Negative log-likelihood of the model given the data and parameters.
    """
    alpha, beta = params
    num_reps = len(choices)
    Pchoice = [0.5] * num_reps
    Q = [0.5, 0.5]
    this_alpha = alpha
    this_beta = beta
    for rep_num in range(num_reps):
        # Calculate probability of choice based on Q-values
        Pchoice[rep_num] = max(0.0001, np.exp(this_beta * Q[choices[rep_num]]) /
                               (np.exp(this_beta * Q[0]) + np.exp(this_beta * Q[1])))

        # Update Q-values with prediction error
        Q[choices[rep_num]] += this_alpha * (rewards[rep_num] - Q[choices[rep_num]])
    return Pchoice
