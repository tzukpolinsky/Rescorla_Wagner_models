import numpy as np


def rescorla_wagner(
        alpha: float,
        beta: float,
        rewards: np.ndarray,
        stimuli_present: np.ndarray = None,
        V_init: float = 0.0
) -> np.ndarray:
    """
    Basic Rescorla-Wagner model implementation for classical conditioning.

    Parameters
    ----------
    alpha : float
        Learning rate parameter (α).
    beta : float
        Salience/associability parameter for the US (β).
    rewards : np.ndarray of shape (n_trials,)
        The actual reward (US) delivered on each trial.
    stimuli_present : np.ndarray of binary array (0 or 1) indicating which stimuli are present on each trial.
        If None, assume a single stimulus present on every trial.
    V_init : float, optional
        Initial associative strength for each stimulus.

    Returns
    -------
    V_history : np.ndarray of shape (n_trials+1, n_stimuli)
        The associative strengths (V) of each stimulus before and after each trial.
        V_history[0, :] is the initial value before any trial.
    """
    # If no stimuli_present provided, assume a single stimulus for every trial
    n_trials = len(rewards)
    n_stimuli = len(stimuli_present)
    # Initialize associative strengths for each stimulus
    V = np.full(n_stimuli, V_init, dtype=float)
    # Store the value of V before and after each trial
    V_history = np.zeros((n_trials + 1, n_stimuli), dtype=float)
    V_history[0] = V
    for t in range(n_trials):
        # 1) Calculate total predicted value on this trial (sum over present stimuli)
        V_sum = np.sum(V * stimuli_present[t])

        # 2) Prediction error = λ (reward) - ΣV_i (for stimuli that are present)
        lambda_t = rewards[t]

        prediction_error = lambda_t - V_sum
        V[t] += alpha * beta * prediction_error
        V_history[t + 1] = V

    return V_history


def example():
    """
    Example usage of the rescorla_wagner function.
    """
    alpha = 0.1
    beta = 1.0

    # Example reward schedule for 10 trials
    rewards = np.array([1, 1, 0, 1, 0, 1, 1, 0, 0, 1])

    # Stimulus presence matrix: shape=(10,2)
    # Stimulus A present on every trial, stimulus B only on odd-numbered trials
    stimuli_present = np.array([1] * len(rewards))

    # Run the Rescorla-Wagner model
    V_history = rescorla_wagner(
        alpha=alpha,
        beta=beta,
        rewards=rewards,
        stimuli_present=stimuli_present,
        V_init=0.0
    )

    # Print the associative strengths for each trial
    for t in range(V_history.shape[0]):
        print(f"Trial {t:2d} => V = {V_history[t]}")


if __name__ == "__main__":
    example()

