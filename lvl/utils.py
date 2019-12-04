"""
Common utility functions.
"""
import numpy as np
import numpy.random as npr
import numba


def lsqn(A, B, transposed=False, rcond=None):
    """
    Returns least-squares solution to an
    under-determined system or least-norm solution
    to and over-determined system A @ X = B.

    Parameters
    ----------
    A : array
        Left hand side

    B : array
        Right hand side

    transposed : bool, optional. Default=False
        If True, considers (A.T @ X) as the left hand
        side and (B.T) as the right hand side of the
        system. Otherwise, (A @ X) and (B) are
        respectively the left and right hand sides.

    rcond : float
        Conditioning parameter passed to np.linalg.lstsq

    Returns
    -------
    X : array
        Solution to the least-squares or least-norm
        problem.
    """
    if transposed:
        return (np.linalg.pinv(A.T) @ B.T).T
    else:
        return np.linalg.pinv(A) @ B


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


@numba.jit(nopython=True)
def min_and_max(arr):
    """
    Returns minimum and maximum of a 1d array, with
    only a single pass.

    Parameters
    ----------
    arr : array-like


    Returns
    -------
    min : arr.dtype
        Value of minimum.

    max : arr.dtype
        Value of maximum.
    """

    minval = arr[0]
    maxval = arr[0]

    for x in arr:
        if x < minval:
            minval = x
        elif x > maxval:
            maxval = x

    return minval, maxval
