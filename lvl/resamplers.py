"""
Methods for shuffling/resampling data to achieve baseline model scores.
"""

import numpy as np

from lvl.utils import rand_orth, get_random_state
from lvl.factor_models import NMF


def destroy_clusters(X, method="rotation", n_components=None, seed=None):
    """
    Resample data, destroying clusters along rows, while
    preserving low-rank structure and norm of the data.
    """

    m, n = X.shape
    rs = get_random_state(seed)

    if method == "rotation":
        return np.dot(rand_orth(m, m), X)

    elif method == "nmf":
        if n_components is None:
            raise ValueError(
                "Method 'nmf' requires 'n_components' "
                "parameter to be specfied."
            )

        # Find low-rank, nonnegative features.
        nc = n_components
        nmf = NMF(nc)
        nmf.fit(X)
        _, H = nmf.factors

        # Resample W uniformly on probability simplex.
        W = rs.dirichlet(np.full(nc, 1 / nc), size=m)

        # Rescale re-sampled data to match norm of X.
        shuff_X = np.dot(W, H)
        return shuff_X * (np.linalg.norm(X) / np.linalg.norm(shuff_X))
