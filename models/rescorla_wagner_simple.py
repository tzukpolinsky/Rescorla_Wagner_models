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

        # 3) Update the associative strengths in one step using NumPy indexing
        #    Only update indices where stimuli_present[t] == 1
        V[stimuli_present[t] == 1] += alpha * beta * prediction_error

        # 4) Store updated V in V_history
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
    stimuli_present = np.array([
        [1, 1],  # Trial 1
        [1, 0],  # Trial 2
        [1, 1],  # Trial 3
        [1, 0],  # Trial 4
        [1, 1],  # Trial 5
        [1, 0],  # Trial 6
        [1, 1],  # Trial 7
        [1, 0],  # Trial 8
        [1, 1],  # Trial 9
        [1, 0]  # Trial 10
    ], dtype=int)

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
