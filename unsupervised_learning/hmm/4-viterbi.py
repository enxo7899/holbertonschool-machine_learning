#!/usr/bin/env python3
"""
Module: 4-viterbi
This module contains the implementation
for Hidden Markov Models.
"""


import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Calculate the most likely sequence of hidden states for
    a Hidden Markov Model using the Viterbi algorithm.

    Args:
        Observation (numpy.ndarray): An array of shape (T,)
        Emission (numpy.ndarray): An array of shape (N, M)
        Transition (numpy.ndarray): An array of shape (N, N)
        Initial (numpy.ndarray): An array of shape (N, 1)

    Returns:
        tuple: A tuple containing the most likely sequence of
        hidden states (list) and its probability (float).
    """
    T = len(Observation)
    N, M = Emission.shape

    if T == 0 or N == 0:
        return None, None

    Viterbi = np.zeros((N, T))
    backpointer = np.zeros((N, T), dtype=int)

    Viterbi[:, 0] = Initial.reshape(-1) * Emission[:, Observation[0]]
    backpointer[:, 0] = -1

    for t in range(1, T):
        for s in range(N):
            max_score = -1
            best_state = -1
            for s_prev in range(N):
                score = Viterbi[s_prev, t - 1] \
                        * Transition[s_prev, s] \
                        * Emission[s, Observation[t]]
                if score > max_score:
                    max_score = score
                    best_state = s_prev

            Viterbi[s, t] = max_score
            backpointer[s, t] = best_state

    path = []
    max_prob_state = np.argmax(Viterbi[:, -1])
    path.append(max_prob_state)
    for t in range(T - 1, 0, -1):
        max_prob_state = backpointer[max_prob_state, t]
        path.append(max_prob_state)

    path.reverse()

    P = np.max(Viterbi[:, -1])

    return path, P


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
    path, P = viterbi(
        Observations,
        Emission,
        Transition,
        Initial.reshape((-1, 1))
    )
    print("Probability of the most likely path:", P)
    print("Most likely path:", path)
