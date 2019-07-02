import numpy as np


def truncated_fourier(Y, n_freq):
    """
    Transforms data using low-frequency Fourier modes.

    Parameters
    ----------
    Y : ndarray
        Matrix of data with shape (m, n)
    n_freq : int
        Number of frequencies to keep. The total number
        of basis functions is 2 * n_freq.

    Returns
    -------
    Yr : ndarray
        Matrix of data with shape (m, n_basis_funcs)
    S : ndarray
        Matrix with shape (n_basis_funcs, n)

    Example
    -------
    >>> Yr, S = truncated_fourier(Y, 10)
    >>> model.fit(Yr)
    >>> U, Vr = model.factors
    >>> model.update(U, np.dot(Vr, S))
    """

    m, n = Y.shape
    tx = np.linspace(-np.pi, np.pi, n + 1)[:-1]

    S = np.empty((n_freq * 2, n))

    i = 0
    for f in range(0, n_freq):
        S[i] = np.sin(tx * (f + 1))
        S[i + 1] = np.cos(tx * (f + 1))
        i += 2

    S /= np.linalg.norm(S, axis=1, keepdims=True)

    Yr = np.dot(Y, np.linalg.pinv(S))

    return Yr, S
