#!/usr/bin/env python3
"""Steady state probabilities of a regular markov chain"""

import numpy as np


def regular(P):
    """
    Function that determines the steady state regular markov chain
    """
    try:
        if len(P.shape) != 2:
            return None

        n = P.shape[0]
        if n != P.shape[1]:
            return None

        # Method by eigendescomposition
        # Formula https://cutt.ly/Ed9Ad7s

        #  (πP).T = π.T ⟹ P.T π.T = π.T (.)
        evals, evecs = np.linalg.eig(P.T)
        """
         break down a matrix into its constituent parts
         where the eigenvectors represent the matrix scales
         and the eigenvalues represent the scaling factors
        """

        # trick: has to be normalized, elements sum to 1
        state = (evecs / evecs.sum())

        # P.T π.T = π.T (.)
        new_state = np.dot(state.T, P)

        # each element of the normalized state vector is greater
        # and if the sum of elements in the vector is close to 1.
        for i in new_state:
            if (i >= 0).all() and np.isclose(i.sum(), 1):
                return i.reshape(1, n)

    except Exception:
        return None
