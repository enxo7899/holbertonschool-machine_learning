#!/usr/bin/env python3
"""
Baum Welch
"""


import numpy as np

def forward(Observations, Emission, Transition, Initial):
    """
    Perform the forward algorithm for a Hidden Markov Model.

    Returns:
        numpy.ndarray: The forward path probabilities of shape (M, T).
    """
    T = len(Observations)
    M, N = Emission.shape

    if T == 0 or M == 0:
        return None

    F = np.zeros((M, T))
    F[:, 0] = Initial[:, 0] * Emission[:, Observations[0]]

    for t in range(1, T):
        for s in range(M):
            F[s, t] = np.sum(F[:, t - 1] * Transition[:, s] * Emission[s, Observations[t]])

    return F

def backward(Observations, Emission, Transition, Initial):
    """
    Perform the backward algorithm for a Hidden Markov Model.

    Returns:
        numpy.ndarray: The backward path probabilities of shape (M, T).
    """
    T = len(Observations)
    M, N = Emission.shape

    if T == 0 or M == 0:
        return None

    B = np.zeros((M, T))
    B[:, -1] = 1.0

    for t in range(T - 2, -1, -1):
        for s in range(M):
            B[s, t] = np.sum(B[:, t + 1] * Transition[s, :] * Emission[:, Observations[t + 1]])

    return B

def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Perform the Baum-Welch algorithm for a Hidden Markov Model.

    Args:
        Observations (numpy.ndarray): An array of shape (T,)
        Transition (numpy.ndarray): An array of shape (M, M)
        Emission (numpy.ndarray): An array of shape (M, N)
        Initial (numpy.ndarray): An array of shape (M, 1)
        iterations (int): The number of times expectation

    Returns:
        tuple: A tuple containing the converged Transition
    """
    M, N = Emission.shape

    for _ in range(iterations):
        # E-step: Calculate forward and backward probabilities
        F = forward(Observations, Emission, Transition, Initial)
        B = backward(Observations, Emission, Transition, Initial)

        # Compute the expected sufficient statistics
        expected_transitions = np.zeros((M, M))
        expected_emissions = np.zeros((M, N))

        for t in range(len(Observations) - 1):
            for i in range(M):
                for j in range(M):
                    expected_transitions[i, j] += F[i, t] * Transition[i, j] * Emission[j, Observations[t + 1]] * B[j, t + 1]

            for i in range(M):
                expected_emissions[i, Observations[t]] += F[i, t] * B[i, t]

        # Normalize the expected statistics
        sum_expected_transitions = np.sum(expected_transitions, axis=1)
        sum_expected_emissions = np.sum(expected_emissions, axis=1)

        for i in range(M):
            Transition[i, :] = expected_transitions[i, :] / sum_expected_transitions[i]
            Emission[i, :] = expected_emissions[i, :] / sum_expected_emissions[i]

    return Transition, Emission
