import numpy as np


def win_stay_lose_switch(epsilon: float, choices: np.ndarray, rewards: np.ndarray) -> np.ndarray:
    """
    Noisy Win-Stay-Lose-Switch model as described in the paper.

    Parameters
    ----------
    epsilon : float
        The overall level of randomness parameter (ε), between 0 and 1.
        With probability 1-ε, the model applies the WSLS rule.
        With probability ε, the model chooses randomly.
    choices : np.ndarray
        Binary array of shape (n_trials,) representing previous choices (0 or 1).
    rewards : np.ndarray
        Binary array of shape (n_trials,) representing rewards (0 or 1).

    Returns
    -------
    probabilities : np.ndarray
        Probability of choosing option 1 in each trial.

    Notes
    -----
    This implements the model described in the paper where:
    - With probability 1-ε, apply win-stay-lose-shift rule
    - With probability ε, choose randomly (50/50)
    - For a two-bandit case, the probability of choosing an option k is:
      p(k) = 1-ε/2 if (previous choice was k and reward=1) OR (previous choice was not k and reward=0)
      p(k) = ε/2 if (previous choice was not k and reward=1) OR (previous choice was k and reward=0)
    """
    if not (0 <= epsilon <= 1):
        raise ValueError("Epsilon parameter must be between 0 and 1.")

    n_trials = len(rewards)
    probabilities = np.zeros(n_trials)

    # For the first trial, we have no previous choice, so set probability to 0.5
    probabilities[0] = 0.5

    for t in range(1, n_trials):
        prev_choice = choices[t - 1]
        prev_reward = rewards[t - 1]

        # Calculate the probability of choosing option 1
        if prev_choice == 1:
            # Previous choice was 1
            if prev_reward == 1:
                # Win-stay: high probability of staying with 1
                probabilities[t] = 1 - epsilon / 2
            else:
                # Lose-switch: low probability of staying with 1
                probabilities[t] = epsilon / 2
        else:
            # Previous choice was 0
            if prev_reward == 1:
                # Win-stay: high probability of staying with 0 (low prob of switching to 1)
                probabilities[t] = epsilon / 2
            else:
                # Lose-switch: high probability of switching to 1
                probabilities[t] = 1 - epsilon / 2

    return probabilities


def example():
    """
    Example usage of the win_stay_lose_switch function.
    """
    # Set epsilon parameter
    epsilon = 0.1

    # Example choices and rewards for 10 trials
    choices = np.array([1, 1, 0, 1, 0, 1, 1, 0, 0, 1])
    rewards = np.array([1, 1, 0, 1, 0, 1, 1, 0, 0, 1])

    # Run the Win-Stay-Lose-Switch model
    probabilities = win_stay_lose_switch(
        epsilon=epsilon,
        choices=choices,
        rewards=rewards
    )

    # Print the choice probabilities for each trial
    for t in range(len(probabilities)):
        if t == 0:
            print(f"Trial {t + 1}: Initial choice probability = {probabilities[t]:.3f}")
        else:
            print(f"Trial {t + 1}: Previous choice = {choices[t - 1]}, Previous reward = {rewards[t - 1]}, "
                  f"Probability of choosing option 1 = {probabilities[t]:.3f}")


if __name__ == "__main__":
    example()
