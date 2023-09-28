#!/usr/bin/env python3
""" Forward Markov chain"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Function that performs the forward algorithm of a HMM
    """
    try:
        # Hidden States
        N = Transition.shape[0]

        # Observations
        T = Observation.shape[0]

        # 1. INITIALIZATION
        # F == alpha
        # initialization α1(j) = πjbj(o1) 1 ≤ j ≤ N
        F = np.zeros((N, T))
        F[:, 0] = Initial.T * Emission[:, Observation[0]]

        # 2. RECURSION
        # formula shorturl.at/amtJT
        # Recursion αt(j) == ∑Ni=1 αt−1(i)ai jbj(ot); 1≤j≤N,1<t≤T
        for t in range(1, T):
            for n in range(N):
                Transitions = Transition[:, n]
                Emissions = Emission[n, Observation[t]]
                F[n, t] = np.sum(Transitions * F[:, t - 1] * Emissions)

        # 3. TERMINATION
        # Termination P(O|λ) == ∑Ni=1 αT (i)
        """
        probabilities represent the likelihood of observing
        the sequence of observations up to time step t
        and being in state j at time step t.
        -1 denotes teh last index, that is time step t
        """
        P = np.sum(F[:, -1])

        return P, F
    except Exception:
        None, None
