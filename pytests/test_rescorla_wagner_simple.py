import numpy as np

from models.rescorla_wagner_simple import rescorla_wagner
import numpy as np
import matplotlib.pyplot as plt
from fitting.minimizing import minimizing_function


def test_single_stimulus():
    # Setup
    alpha = 0.1
    beta = 1.0
    rewards = np.array([1, 1, 0, 1, 0])
    stimuli_present = np.array([1] * len(rewards))
    V_history = rescorla_wagner(alpha, beta, rewards, stimuli_present, V_init=0)
    final_v = V_history[-1, 0]  # last trial's V value

    assert np.isclose(final_v, 0.06875, atol=1e-5), \
        f"Expected ~0.22851, got {final_v}"





if __name__ == "__main__":
    test_single_stimulus()