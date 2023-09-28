#!/usr/bin/env python3
"""
Backwards Task
"""


import numpy as np

def backward(Observation, Emission, Transition, Initial):
    """
    Perform the backward algorithm for a Hidden Markov Model.

    Args:
        Observation (numpy.ndarray): An array of shape (T,) containing the indices of observations.
        Emission (numpy.ndarray): An array of shape (N, M) containing the emission probabilities.
        Transition (numpy.ndarray): An array of shape (N, N) containing the transition probabilities.
        Initial (numpy.ndarray): An array of shape (N, 1) containing the initial probabilities.

    Returns:
        tuple: A tuple containing the likelihood of the observations given the model (float)
        and the backward path probabilities (numpy.ndarray) of shape (N, T).
    """
    T = len(Observation)
    N, M = Emission.shape

    if T == 0 or N == 0:
        return None, None

    B = np.zeros((N, T))
    B[:, -1] = 1.0

    for t in range(T - 2, -1, -1):
        for s in range(N):
            B[s, t] = np.sum(B[:, t + 1] * Transition[s, :] * Emission[:, Observation[t + 1]])

    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])

    return P, B

# Example usage
if __name__ == '__main__':
    np.random.seed(1)
    Emission = np.array([[0.90, 0.10, 0.00, 0.00, 0.00, 0.00],
                         [0.40, 0.50, 0.10, 0.00, 0.00, 0.00],
                         [0.00, 0.25, 0.50, 0.25, 0.00, 0.00],
                         [0.00, 0.00, 0.05, 0.70, 0.15, 0.10],
                         [0.00, 0.00, 0.00, 0.20, 0.50, 0.30]])
    Transition = np.array([[0.60, 0.39, 0.01, 0.00, 0.00],
                           [0.20, 0.50, 0.30, 0.00, 0.00],
                           [0.01, 0.24, 0.50, 0.24, 0.01],
                           [0.00, 0.00, 0.15, 0.70, 0.15],
                           [0.00, 0.00, 0.01, 0.39, 0.60]])
    Initial = np.array([0.05, 0.20, 0.50, 0.20, 0.05])
    Hidden = [np.random.choice(5, p=Initial)]
    for _ in range(364):
        Hidden.append(np.random.choice(5, p=Transition[Hidden[-1]]))
    Hidden = np.array(Hidden)
    Observations = []
    for s in Hidden:
        Observations.append(np.random.choice(6, p=Emission[s]))
    Observations = np.array(Observations)
    P, B = backward(Observations, Emission, Transition, Initial.reshape((-1, 1)))
    print(P)
    print(B)
