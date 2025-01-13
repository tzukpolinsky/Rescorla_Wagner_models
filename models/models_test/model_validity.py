import numpy as np
from matplotlib import pyplot as plt

from fitting.minimizing import minimizing_function
from models.rescorla_wagner_simple import rescorla_wagner


def generate_synthetic_data(alpha, beta, n_trials=100):
    rewards = np.random.choice([0, 1], size=n_trials, p=[0.5, 0.5])
    stimuli_present = np.array([1] * len(rewards))  # Single stimulus present in each trial
    V_history = rescorla_wagner(alpha, beta, rewards, stimuli_present).reshape(-1)[:n_trials]
    # Generate binary choices based on the associative strength and a softmax choice
    choices = np.random.binomial(1, 1 / (1 + np.exp(-beta * (V_history)))).flatten()
    return rewards, stimuli_present, choices


# Fit the synthetic data to recover parameters
def fit_synthetic_data(rewards, stimuli_present, observed_choices, initial_guess=[0.5, 0.5]):
    minimize_options = {
        'method': 'L-BFGS-B',
        'options': {'disp': False},
        'bounds': [(0.01, 1.0), (0.1, 10.0)]
    }
    result = minimizing_function(
        model_function=rescorla_wagner,
        initial_parameters=initial_guess,
        rewards=rewards,
        stimuli_present=stimuli_present,
        observed_data=observed_choices,
        cost_metric='log-likelihood',
        minimize_options=minimize_options
    )
    return result.x  # Return recovered parameters


# Perform parameter recovery test
def parameter_recovery_test_rescorla_wagner(n_tests=20, n_trials=100):
    true_params = []
    recovered_params = []

    for _ in range(n_tests):
        alpha_true = np.random.uniform(0.1, 1)  # True alpha (learning rate)
        beta_true = np.random.uniform(1.0, 5.0)  # True beta (inverse temperature)
        true_params.append((alpha_true, beta_true))

        # Generate synthetic data
        rewards, stimuli_present, choices = generate_synthetic_data(alpha_true, beta_true, n_trials)

        # Fit the synthetic data to recover parameters
        recovered_alpha, recovered_beta = fit_synthetic_data(rewards, stimuli_present, choices,
                                                             initial_guess=[
                                                                 np.random.uniform(max(0.1, alpha_true - 0.2),
                                                                                   alpha_true + 0.2),
                                                                 np.random.uniform(beta_true - 1, beta_true + 1)])
        recovered_params.append((recovered_alpha, recovered_beta))

    true_params = np.array(true_params)
    recovered_params = np.array(recovered_params)

    # Scatter plot to compare true vs recovered parameters
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

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parameter_recovery_test_rescorla_wagner(n_tests=100,n_trials=60)
