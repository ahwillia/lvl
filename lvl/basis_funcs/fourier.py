import numpy as np
import numbers
import itertools


def truncated_fourier(n_freq, shape):
    """
    Transforms data using low-frequency Fourier modes.

    Parameters
    ----------
    n_freq : int
        Number of frequencies to keep. The total number
        of basis functions is 2 * (n + 1) ** len(shape) - 1.
    shape : tuple
        Number of samples along each dimension.

    Returns
    -------
    Yr : ndarray
        Matrix of data with shape (m, n_basis_funcs)
    S : ndarray
        Matrix with shape (n_basis_funcs, n)

    Reference
    ---------
    G.D. Konidaris, S. Osentoski and P.S. Thomas. Value
    Function Approximation in Reinforcement Learning using
    the Fourier Basis. In Proceedings of the Twenty-Fifth
    Conference on Artificial Intelligence, pages 380-385,
    August 2011.

    Example
    -------
    >>> m, n = shape(data)
    >>> F = truncated_fourier(10, n)
    >>> model.fit(data, basis=F)
    >>> U, Vr = model.factors
    >>> V = np.dot(Vr, F)
    >>> model.update(U, V)
    """

    if isinstance(shape, numbers.Integral):
        shape = (shape,)

    gs = [np.linspace(0, np.pi, n + 1)[:-1] for n in shape[::-1]]
    Gs = np.meshgrid(*gs)
    gxs = np.array([g.ravel() for g in Gs])

    # Allocate array for basis functions.
    S = np.empty((2 * n_freq ** len(shape) - 1, np.prod(shape)))

    # First basis function is constant.
    S[0] = 1.0

    # Additional basis functions come in sine-cosine pairs.
    i = 1
    for c in itertools.product(*[range(n_freq) for n in shape]):

        if np.sum(c) == 0:
            continue  # constant term is already included.

        ca = np.array(c)[:, None]
        S[i] = np.sin(np.sum(gxs * ca, axis=0))
        S[i + 1] = np.cos(np.sum(gxs * ca, axis=0))
        i += 2

    print(i)
    print(len(S))

    # Normalize rows before returning
    return S / np.linalg.norm(S, axis=1, keepdims=True)
