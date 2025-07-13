import numpy as np
from matplotlib import pyplot as plt

import pyddm
from pyddm import gddm, Sample
import pandas as pd
from computational_models.random_response import random_response
from fitting.minimizing import minimizing_rescorla_wagner_model, minimizing_reinforcement_learning_model, \
    minimizing_random_response, minimizing_win_stay_lose_switch, fit_ddm
from computational_models.reinforcement_learning_based_choice_model import reinforcement_learning_simple_model
from computational_models.rescorla_wagner_simple import rescorla_wagner
from computational_models.win_stay_lose_switch import win_stay_lose_switch


def generate_synthetic_data_win_stay_lose_switch(epsilon, n_trials=100):
    """
    Generate synthetic data for the Win-Stay-Lose-Switch model.

    Parameters
    ----------
    epsilon : float
        The randomness parameter (epsilon) for the WSLS model.
    n_trials : int, optional
        Number of trials to simulate (default is 100).

    Returns
    -------
    choices : np.ndarray
        Binary array of simulated choices.
    rewards : np.ndarray
        Binary array of rewards.
    """
    if not (0 <= epsilon <= 1):
        raise ValueError("Epsilon parameter must be between 0 and 1.")

    # Initialize arrays
    choices = np.zeros(n_trials, dtype=int)
    rewards = np.random.choice([0, 1], size=n_trials, p=[0.5, 0.5])  # Random rewards

    # First choice is random
    choices[0] = np.random.choice([0, 1])

    # Generate choices based on WSLS model
    for t in range(1, n_trials):
        prev_choice = choices[t - 1]
        prev_reward = rewards[t - 1]

        # Calculate probability of choosing option 1
        if prev_choice == 1:
            if prev_reward == 1:
                # Win-stay: high probability of staying with 1
                p_choose_1 = 1 - epsilon / 2
            else:
                # Lose-switch: low probability of staying with 1
                p_choose_1 = epsilon / 2
        else:
            if prev_reward == 1:
                # Win-stay: high probability of staying with 0 (low prob of switching to 1)
                p_choose_1 = epsilon / 2
            else:
                # Lose-switch: high probability of switching to 1
                p_choose_1 = 1 - epsilon / 2

        # Generate choice based on probability
        choices[t] = np.random.binomial(1, p_choose_1)

    return choices, rewards


def fit_synthetic_data_win_stay_lose_switch(choices, rewards, initial_guess=[0.5]):
    """
    Fit the Win-Stay-Lose-Switch model to synthetic data.

    Parameters
    ----------
    choices : np.ndarray
        Binary array of choices.
    rewards : np.ndarray
        Binary array of rewards.
    initial_guess : list, optional
        Initial guess for the epsilon parameter.

    Returns
    -------
    recovered_epsilon : float
        The recovered epsilon parameter value.
    """
    minimize_options = {
        'method': 'L-BFGS-B',
        'bounds': [(0.01, 0.99)],  # Slightly narrowed bounds to avoid edge cases
        'options': {'disp': False}
    }

    result = minimizing_win_stay_lose_switch(
        model_function=win_stay_lose_switch,
        initial_parameters=initial_guess,
        choices=choices,
        rewards=rewards,
        cost_metric='log-likelihood',
        minimize_options=minimize_options
    )

    return result.x[0]  # Return the recovered epsilon


def parameter_recovery_test_win_stay_lose_switch(n_tests=20, n_trials=100):
    """
    Test parameter recovery for the Win-Stay-Lose-Switch model.

    Parameters
    ----------
    n_tests : int, optional
        Number of parameter recovery tests to perform.
    n_trials : int, optional
        Number of trials in each simulated dataset.

    Returns
    -------
    true_params : np.ndarray
        True epsilon parameter values.
    recovered_params : np.ndarray
        Recovered epsilon parameter values.
    """
    true_params = []
    recovered_params = []

    for _ in range(n_tests):
        # Generate random true parameter
        epsilon_true = np.random.uniform(0.05, 0.95)
        true_params.append(epsilon_true)

        # Generate synthetic data
        choices, rewards = generate_synthetic_data_win_stay_lose_switch(epsilon_true, n_trials)

        # Initial guess (random near the true value)
        initial_guess = [max(0.01, min(0.99, epsilon_true + np.random.uniform(-0.2, 0.2)))]

        # Fit the model
        recovered_epsilon = fit_synthetic_data_win_stay_lose_switch(choices, rewards, initial_guess)
        recovered_params.append(recovered_epsilon)

    # Convert to numpy arrays
    true_params = np.array(true_params)
    recovered_params = np.array(recovered_params)

    # Plot results
    plt.figure(figsize=(8, 8))
    plt.scatter(true_params, recovered_params, alpha=0.7, label="Recovered Parameters")
    plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Ideal Recovery (y=x)")

    plt.title(f"Win-Stay-Lose-Switch Parameter Recovery\nN trials: {n_trials}, N tests: {n_tests}")
    plt.xlabel("True Epsilon (ε)")
    plt.ylabel("Recovered Epsilon (ε)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.show()

    return true_params, recovered_params


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
    stimuli_present = np.ones((n_trials, n_stimuli), dtype=int)  # Single stimulus present in each trial
    V_history, stimuli_present_output = rescorla_wagner(alpha, beta, rewards, stimuli_present)
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


def generate_synthetic_data_ddm(driftrate, boundary, starting_point, ndt, n_trials=100):
    """
    Generate synthetic data from a Drift Diffusion Model (DDM) with known parameters.

    Parameters
    ----------
    driftrate : float
        The drift rate parameter (speed of evidence accumulation).
    boundary : float
        The boundary separation parameter (decision threshold).
    starting_point : float
        The starting point parameter (bias toward one response).
    ndt : float
        The non-decision time parameter (encoding + motor response time).
    n_trials : int, optional
        Number of trials to simulate (default is 100).

    Returns
    -------
    rt_data : np.ndarray
        Array of reaction times.
    response_data : np.ndarray
        Binary array of responses (0 or 1).
    """

    # Create a DDM model with the specified parameters
    model = gddm(drift=driftrate, noise=1, bound=boundary, starting_position=starting_point, nondecision=ndt,dt=0.005,  # Default is 0.01, try 0.005 or 0.001
    dx=0.005)
    # Simulate data from the model
    solution = model.solve()
    sample = solution.sample(n_trials)
    df_sample = sample.to_pandas_dataframe(drop_undecided=True)
    # Extract reaction times and responses
    rt_data = df_sample['RT'].values
    response_data = df_sample['choice'].values

    return rt_data, response_data


def fit_synthetic_data_ddm(rt_data, response_data, parameter_ranges=None):
    """
    Fit a DDM model to synthetic data to recover parameters.

    Parameters
    ----------
    rt_data : np.ndarray
        Array of reaction times.
    response_data : np.ndarray
        Binary array of responses (0 or 1).
    parameter_ranges : dict, optional
        Dictionary of parameter ranges for the model. If None, default ranges are used.

    Returns
    -------
    recovered_params : dict
        Dictionary of recovered parameters.
    """
    # Fit the model using the fit_ddm function
    fitted_model = fit_ddm(rt_data, response_data, parameter_ranges)

    # Extract the recovered parameters
    recovered_params = fitted_model.parameters()

    return recovered_params


def plot_ddm_parameter_recovery(true_params, recovered_params, title):
    """
    Visualize the parameter recovery results for the DDM model.

    Parameters
    ----------
    true_params : list of tuples
        List of true parameter values.
    recovered_params : list of tuples
        List of recovered parameter values.
    title : str
        Title for the plot.
    """
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))

    # Extract parameters
    true_driftrate = [p[0] for p in true_params]
    true_boundary = [p[1] for p in true_params]
    true_starting_point = [p[2] for p in true_params]
    true_ndt = [p[3] for p in true_params]

    recovered_driftrate = [float(p['drift']['drift']) for p in recovered_params]
    recovered_boundary = [float(p['bound']['B']) for p in recovered_params]
    recovered_starting_point = [float(p['IC']['x0']) for p in recovered_params]
    recovered_ndt = [float(p['overlay']['nondectime']) for p in recovered_params]

    # Plot driftrate recovery
    ax[0, 0].scatter(true_driftrate, recovered_driftrate, label="Drift Rate", color='blue', alpha=0.7)
    ax[0, 0].plot([-10, 10], [-10, 10], color='gray', linestyle='--', label="Ideal Recovery Line")
    ax[0, 0].set_title("A: Drift Rate Recovery")
    ax[0, 0].set_xlabel("Simulated Drift Rate (true)")
    ax[0, 0].set_ylabel("Recovered Drift Rate (fit)")
    ax[0, 0].legend()

    # Plot boundary recovery
    ax[0, 1].scatter(true_boundary, recovered_boundary, label="Boundary", color='orange', alpha=0.7)
    ax[0, 1].plot([0, 5], [0, 5], color='gray', linestyle='--', label="Ideal Recovery Line")
    ax[0, 1].set_title("B: Boundary Recovery")
    ax[0, 1].set_xlabel("Simulated Boundary (true)")
    ax[0, 1].set_ylabel("Recovered Boundary (fit)")
    ax[0, 1].legend()

    # Plot starting point recovery
    ax[1, 0].scatter(true_starting_point, recovered_starting_point, label="Starting Point", color='green', alpha=0.7)
    ax[1, 0].plot([-0.5, 0.5], [-0.5, 0.5], color='gray', linestyle='--', label="Ideal Recovery Line")
    ax[1, 0].set_title("C: Starting Point Recovery")
    ax[1, 0].set_xlabel("Simulated Starting Point (true)")
    ax[1, 0].set_ylabel("Recovered Starting Point (fit)")
    ax[1, 0].legend()

    # Plot non-decision time recovery
    ax[1, 1].scatter(true_ndt, recovered_ndt, label="Non-Decision Time", color='purple', alpha=0.7)
    ax[1, 1].plot([0, 0.5], [0, 0.5], color='gray', linestyle='--', label="Ideal Recovery Line")
    ax[1, 1].set_title("D: Non-Decision Time Recovery")
    ax[1, 1].set_xlabel("Simulated Non-Decision Time (true)")
    ax[1, 1].set_ylabel("Recovered Non-Decision Time (fit)")
    ax[1, 1].legend()

    for i in range(2):
        for j in range(2):
            for spine in ['top', 'right']:
                ax[i, j].spines[spine].set_visible(False)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def parameter_recovery_test_ddm(n_tests=20, n_trials=100):
    """
    Test parameter recovery for the Drift Diffusion Model (DDM).

    Parameters
    ----------
    n_tests : int, optional
        Number of parameter recovery tests to perform.
    n_trials : int, optional
        Number of trials in each simulated dataset.

    Returns
    -------
    true_params : list of tuples
        True parameter values.
    recovered_params : list of tuples
        Recovered parameter values.
    """
    true_params = []
    recovered_params = []

    for _ in range(n_tests):
        # Generate random true parameters
        bias = 0.1
        driftrate_true = np.random.uniform(-3, 3)  # Drift rate
        boundary_true = np.random.uniform(0.5, 2)  # Boundary
        starting_point_true = np.random.uniform(-0.5, 0.5)  # Starting point
        ndt_true = np.random.uniform(0.1, 0.3)  # Non-decision time

        true_params.append((driftrate_true, boundary_true, starting_point_true, ndt_true))

        # Generate synthetic data
        rt_data, response_data = generate_synthetic_data_ddm(
            driftrate=driftrate_true,
            boundary=boundary_true,
            starting_point=starting_point_true,
            ndt=ndt_true,
            n_trials=n_trials
        )

        # Define parameter ranges for fitting
        parameter_ranges = {
            "driftrate": (-3, 3),
            "B": (0.5, 2),
            "x0": (-0.5, 0.5),
            "ndt": (0.1, 0.3)
        }

        # Fit the synthetic data to recover parameters
        recovered_params_dict = fit_synthetic_data_ddm(rt_data, response_data, parameter_ranges)
        recovered_params.append(recovered_params_dict)

    # Plot the parameter recovery results
    plot_ddm_parameter_recovery(true_params, recovered_params,
                               f"Drift Diffusion Model Parameter Recovery\nN trials: {n_trials}, N tests: {n_tests}")

    return true_params, recovered_params


if __name__ == "__main__":
    # Test only the DDM parameter recovery with a small number of tests and trials
    parameter_recovery_test_ddm(n_tests=100, n_trials=60)

    # Uncomment to run all parameter recovery tests
    # parameter_recovery_test_win_stay_lose_switch(n_tests=100, n_trials=60)
    # parameter_recovery_test_random_response(n_tests=100, n_trials=60)
    # parameter_recovery_test_rescorla_wagner(n_tests=100, n_trials=60)
    # parameter_recovery_test_reinforcement_learning(n_tests=100, n_trials=60)
    # parameter_recovery_test_ddm(n_tests=100, n_trials=60)
