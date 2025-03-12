import numpy as np
from matplotlib import pyplot as plt

from Rescorla_models.random_response import random_response
from fitting.minimizing import minimizing_rescorla_wagner_model, minimizing_reinforcement_learning_model, \
    minimizing_random_response
from Rescorla_models.reinforcement_learning_based_choice_model import reinforcement_learning_simple_model
from Rescorla_models.rescorla_wagner_simple import rescorla_wagner


def generate_synthetic_data_random_responding(b, n_trials=100):
    """
    Generate synthetic data for the Random Responding model.

    Parameters
    ----------
    b : float
        Bias parameter. Probability of responding "1".
    n_trials : int, optional
        Number of trials to simulate (default is 100).

    Returns
    -------
    rewards : np.ndarray
        Binary array of rewards (0 or 1) for each trial.
    observed_data : np.ndarray
        Simulated binary responses (0 or 1) based on the bias parameter and rewards.
    """
    if not (0 <= b <= 1):
        raise ValueError("Bias parameter b must be between 0 and 1.")

    # Generate rewards randomly with a 50% chance for 0 or 1
    rewards = np.random.choice([0, 1], size=n_trials, p=[0.5, 0.5])

    # Calculate probabilities (pchoice) based on the bias parameter b
    pchoice = random_response(b, rewards)

    # Generate observed data using binomial sampling from pchoice
    observed_data = np.random.binomial(1, pchoice)

    return rewards, observed_data


def generate_synthetic_data_reinforcement_learning(alpha, beta, n_trials=100):
    """
    Generate synthetic data for the reinforcement learning simple model.
    """
    choices = np.random.choice([0, 1], size=n_trials)  # Random initial choices
    rewards = np.random.choice([0, 1], size=n_trials, p=[0.5, 0.5])  # Rewards based on 50% chance
    Pchoice = reinforcement_learning_simple_model([alpha, beta], choices, rewards)  # Model predictions

    # Generate binary observed choices based on probabilities
    observed_choices = np.random.binomial(1, Pchoice)
    return rewards, choices, observed_choices


def generate_synthetic_data_rescorla_wagner(alpha, beta, n_trials=100):
    rewards = np.random.choice([0, 1], size=n_trials, p=[0.5, 0.5])
    n_trials = len(rewards)
    n_stimuli = 1
    stimuli_present = np.ones((n_trials, n_stimuli), dtype=int) # Single stimulus present in each trial
    V_history,stimuli_present_output = rescorla_wagner(alpha, beta, rewards, stimuli_present)
    V_present = np.sum(V_history[1:] * stimuli_present_output, axis=1)
    # Generate binary choices based on the associative strength and a softmax choice
    choices = np.random.binomial(1, 1 / (1 + np.exp(-beta * (V_present)))).flatten()
    return rewards, stimuli_present, choices


def fit_synthetic_data_random_response(rewards, observed_choices, initial_guess=[0.7]):
    minimize_options = {
        'method': 'L-BFGS-B',
        'options': {'disp': False},
        'bounds': [(0.01, 1.0)]  # Bounds for alpha and beta
    }
    result = minimizing_random_response(
        model_function=random_response,
        initial_parameters=initial_guess,
        rewards=rewards,
        observed_data=observed_choices,
        cost_metric='log-likelihood',
        minimize_options=minimize_options
    )
    return result.x  # Return recovered parameters


# Fit the synthetic data to recover parameters
def fit_synthetic_data_rescorla_wagner(rewards, stimuli_present, observed_choices, initial_guess=[0.5, 0.5]):
    minimize_options = {
        'method': 'L-BFGS-B',
        'options': {'disp': False},
        'bounds': [(0.01, 1.0), (0.1, 10.0)]
    }
    result = minimizing_rescorla_wagner_model(
        model_function=rescorla_wagner,
        initial_parameters=initial_guess,
        rewards=rewards,
        stimuli_present=stimuli_present,
        observed_data=observed_choices,
        cost_metric='log-likelihood',
        minimize_options=minimize_options
    )
    return result.x  # Return recovered parameters


def fit_synthetic_data_reinforcement_learning(rewards, choices, observed_choices, initial_guess=[0.5, 0.5]):
    minimize_options = {
        'method': 'L-BFGS-B',
        'options': {'disp': False},
        'bounds': [(0.01, 1.0), (0.1, 10.0)]  # Bounds for alpha and beta
    }
    result = minimizing_reinforcement_learning_model(
        model_function=reinforcement_learning_simple_model,
        initial_parameters=initial_guess,
        rewards=rewards,
        choices=choices,
        observed_data=observed_choices,
        cost_metric='log-likelihood',
        minimize_options=minimize_options
    )
    return result.x  # Return recovered parameters


def parameter_recovery_test_random_response(n_tests=20, n_trials=100):
    true_params = []
    recovered_params = []

    for _ in range(n_tests):
        b_true = np.random.uniform(0.5, 1.0)  # True alpha (learning rate)
        true_params.append(b_true)

        # Generate synthetic data
        rewards, observed_choices = generate_synthetic_data_random_responding(b_true, n_trials)

        # Fit the synthetic data to recover parameters
        initial_guess = [np.random.uniform(max(0.1, b_true - 0.2), b_true + 0.2)]
        recovered_b = fit_synthetic_data_random_response(rewards, observed_choices, initial_guess)
        recovered_params.append(recovered_b)

    true_params = np.array(true_params)
    recovered_params = np.array(recovered_params)
    plt.figure(figsize=(8, 8))
    plt.scatter(true_params, recovered_params, alpha=0.7, label="Recovered Parameters")
    plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Ideal Recovery (y=x)")

    plt.title("Parameter Recovery Test For Random Response")
    plt.xlabel("True Parameters")
    plt.ylabel("Recovered Parameters")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.show()
    return true_params, recovered_params


def parameter_recovery_test_reinforcement_learning(n_tests=20, n_trials=100):
    true_params = []
    recovered_params = []

    for _ in range(n_tests):
        alpha_true = np.random.uniform(0.1, 1.0)  # True alpha (learning rate)
        beta_true = np.random.uniform(0.1, 5.0)  # True beta (inverse temperature)
        true_params.append((alpha_true, beta_true))

        # Generate synthetic data
        rewards, choices, observed_choices = generate_synthetic_data_reinforcement_learning(alpha_true, beta_true,
                                                                                            n_trials)

        # Fit the synthetic data to recover parameters
        initial_guess = [np.random.uniform(max(0.1, alpha_true - 0.2), alpha_true + 0.2),
                         np.random.uniform(max(0.1, beta_true - 1.0), beta_true + 1.0)]
        recovered_alpha, recovered_beta = fit_synthetic_data_reinforcement_learning(rewards, choices, observed_choices,
                                                                                    initial_guess)
        recovered_params.append((recovered_alpha, recovered_beta))

    true_params = np.array(true_params)
    recovered_params = np.array(recovered_params)
    plot_parameter_recovery(true_params, recovered_params,
                            f"Reinforcement Learning Model Parameter Recovery\nN trials: {n_trials}, N tests: {n_tests}")

    return true_params, recovered_params


def plot_parameter_recovery(true_params, recovered_params, title):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Subplot A: Learning rate (alpha) recovery (linear-linear scale)
    ax[0].scatter(true_params[:, 0], recovered_params[:, 0], label="Alpha (learning rate)", color='blue', alpha=0.7)
    ax[0].plot([0, 1], [0, 1], color='gray', linestyle='--', label="Ideal Recovery Line")
    ax[0].set_title("A: Learning Rate (Alpha) Recovery")
    ax[0].set_xlabel("Simulated Alpha (true)")
    ax[0].set_ylabel("Recovered Alpha (fit)")
    ax[0].legend()
    ax[0].set_xlim([0, 1])
    ax[0].set_ylim([0, 1])

    # Subplot B: Softmax temperature (beta) recovery (log-log scale)
    ax[1].scatter(true_params[:, 1], recovered_params[:, 1], label="Beta (inverse temperature)", color='orange',
                  alpha=0.7)
    ax[1].plot([1, 10], [1, 10], color='gray', linestyle='--', label="Ideal Recovery Line")
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_title("B: Softmax Temperature (Beta) Recovery (Log-Log)")
    ax[1].set_xlabel("Simulated Beta (true)")
    ax[1].set_ylabel("Recovered Beta (fit)")
    ax[1].legend()
    ax[1].set_xlim([1, 10])
    ax[1].set_ylim([1, 10])
    for i in range(2):
        for spine in ['top', 'right']:
            ax[i].spines[spine].set_visible(False)
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def parameter_recovery_test_rescorla_wagner(n_tests=20, n_trials=100):
    true_params = []
    recovered_params = []

    for _ in range(n_tests):
        alpha_true = np.random.uniform(0.1, 1)  # True alpha (learning rate)
        beta_true = np.random.uniform(1.0, 5.0)  # True beta (inverse temperature)
        true_params.append((alpha_true, beta_true))

        # Generate synthetic data
        rewards, stimuli_present, choices = generate_synthetic_data_rescorla_wagner(alpha_true, beta_true, n_trials)

        # Fit the synthetic data to recover parameters
        recovered_alpha, recovered_beta = fit_synthetic_data_rescorla_wagner(rewards, stimuli_present, choices,
                                                                             initial_guess=[
                                                                                 np.random.uniform(
                                                                                     max(0.1, alpha_true - 0.2),
                                                                                     alpha_true + 0.2),
                                                                                 np.random.uniform(beta_true - 1,
                                                                                                   beta_true + 1)])
        recovered_params.append((recovered_alpha, recovered_beta))

    true_params = np.array(true_params)
    recovered_params = np.array(recovered_params)
    plot_parameter_recovery(true_params, recovered_params,
                            f"Rescorla-Wagner Model Parameter Recovery\nN trials: {n_trials}, N tests: {n_tests}")

    return true_params, recovered_params


if __name__ == "__main__":
    parameter_recovery_test_random_response(n_tests=100,n_trials=60)
    parameter_recovery_test_rescorla_wagner(n_tests=100, n_trials=60)
    parameter_recovery_test_reinforcement_learning(n_tests=100, n_trials=60)
