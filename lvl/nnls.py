from scipy.optimize import nnls as scipy_nnls

import numpy as np
import numba


def nnls(A, B, method="cd", max_iter=100, tol=1e-9):
    """
    Solve nonnegative least squares problem
    ``argmin_x || Ax - b ||_2`` for ``x>=0``.

    Parameters
    ----------
    A : ndarray
        Matrix ``A`` as shown above.
    b : ndarray
        Right-hand side vector.
    method : str
        Specifies optimization method. Options are
        'cd' (coordinate descent), 'scipy'
        (scipy.optimize.nnls).
    maxiter: int
        Maximum number of iterations.
    tol : float
        Stopping tolerance.

    Returns
    -------
    x : ndarray
        Solution vector.
    """

    # Check inputs.
    B = B[:, None] if B.ndim == 1 else B

    if (A.ndim != 2) or (B.ndim != 2):
        raise ValueError("Expected matrix inputs.")

    if A.shape[0] != B.shape[0]:
        raise ValueError("Incompatible dimensions.")

    # Dispatch desired method.
    if method == "cd":
        return cd_nnls(A, B, max_iter, tol)

    if method == "scipy":
        x = [scipy_nnls(A, b)[0] for b in B.T]
        return np.array(x).T

    else:
        raise ValueError("Did not recognize method.")


@numba.jit(nopython=True, parallel=True)
def cd_nnls(A, B, max_iter, tol):
    """
    Solve ``argmin_x || Ax - b ||_2`` for ``x>=0``

    Parameters
    ----------
    A : ndarray
        Matrix ``A`` as shown above.
    b : ndarray
        Right-hand side vector.
    maxiter: int
        Maximum number of iterations.
    tol : float
        Stopping tolerance.

    Returns
    -------
    x : ndarray
        Solution vector.
    """

    AtA = np.dot(A.T, A)
    AtB = np.dot(A.T, B)
    X = np.zeros((A.shape[1], B.shape[1]))

    for k in numba.prange(B.shape[1]):
        cd_quad_prog(AtA, AtB[:, k], X[:, k], max_iter, tol)

    return X


@numba.jit(nopython=True, parallel=True)
def cd_quad_prog(H, c, x, max_iter, tol):
    """
    Solves unconstrained quadratic programming problem
    defined by positive definite matrix H and vector c.
    Finds x that minimizes:

        0.5 * x.T @ H @ x - x.T @ c

    Parameters
    ----------
    H : ndarray
        Positive definite n x n matrix.
    c : ndarray
        n-dimensional vector.
    x : ndarray
        n-dimensional vector, overwritten to store result.
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance.
    """

    # Initialize gradient.
    n = H.shape[0]
    x.fill(0.0)
    grad = -c

    for i in range(max_iter):

        # Tracks total parameter update.
        total_dx = 0.0

        # Update all x.
        for j in range(n):

            # Update coordinate x[j]
            new_x = np.maximum(0.0, x[j] - (grad[j] / H[j, j]))
            dx = new_x - x[j]

            # If x[j] changed, update gradient.
            if dx != 0.0:
                x[j] = new_x
                grad += dx * H[j]

            # Keep track of total absolute changes.
            total_dx += dx ** 2

        # Check convergence
        if np.sqrt(total_dx) < (np.sqrt(n) * tol):
            break
