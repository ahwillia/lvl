"""
Common utility functions.
"""
import numpy as np
import numpy.random as npr


def get_random_state(seed_or_rs):
    """
    Converts an integer to a seeded RandomState instance.
    If input is already a RandomState instance, it returns
    it unchanged.
    """

    if isinstance(seed_or_rs, npr.RandomState):
        return seed_or_rs
    else:
        return npr.RandomState(seed_or_rs)


def rand_orth(m, n=None, seed=None):
    """
    Creates a random matrix with orthogonal columns or rows.

    Parameters
    ----------
    m : int
        First dimension
    n : int
        Second dimension (if None, matrix is m x m)

    Returns
    -------
    Q : ndarray
        An m x n random matrix. If m > n, the columns are orthonormal.
        If m < n, the rows are orthonormal. If m == n, the result is
        an orthogonal matrix.
    """
    rs = get_random_state(seed)
    n = m if n is None else n
    if n > m:
        return np.linalg.qr(rs.randn(n, m))[0].T
    else:
        return np.linalg.qr(rs.randn(m, n))[0]
