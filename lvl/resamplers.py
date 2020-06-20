"""
Methods for shuffling/resampling data to achieve baseline model scores.
"""

import numpy as np

from lvl.utils import rand_orth, get_random_state
from lvl.factor_models import NMF


class PermuteEachRow:
    """
    Resamples matrix data by randomly permuting
    each row.
    """

    def __init__(self, seed=None):
        self._rs = get_random_state(seed)

    def __call__(self, X):
        Y = np.copy(X)
        m, n = Y.shape
        for i in range(m):
            Y[i] = Y[i][self._rs.permutation(n)]
        return Y


class RotationResampler:
    """
    Resamples mean-centered data X by Q @ X
    for random rotation matrix Q.
    """

    def __init__(self, seed=None):
        self._rs = get_random_state(seed)

    def __call__(self, X):
        n = X.shape[0]
        Q = rand_orth(n, n, seed=self._rs)
        m = np.mean(X, axis=0, keepdims=True)
        return (Q @ (X - m)) + m


class MaxEntResampler:
    """
    Fits multivariate Gaussian to rows of
    X and samples new data from this. This
    is the maximum entropy distribution
    constrained by the first two empirical
    moments of the data.
    """

    def __init__(self, seed=None):
        self._rs = get_random_state(seed)

    def __call__(self, X):
        m = np.mean(X, axis=0)
        Xc = X - m[None, :]
        S = (Xc.T @ Xc) / X.shape[0]
        return self._rs.multivariate_normal(m, S, size=X.shape[0])


class NMFResampler:

    def __init__(self, n_components, seed=None):
        self.nc = n_components
        self._rs = get_random_state(seed)

    def __call__(self, X):
        nmf = NMF(nc)
        nmf.fit(X)
        _, H = nmf.factors

        # Resample W uniformly on probability simplex.
        W = rs.dirichlet(np.full(nc, 1 / nc), size=m)

        # Rescale re-sampled data to match norm of X.
        shuff_X = np.dot(W, H)
        return shuff_X * (np.linalg.norm(X) / np.linalg.norm(shuff_X))
