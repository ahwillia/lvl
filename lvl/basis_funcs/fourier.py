import numpy as np
import itertools


class TruncatedFourier:
    """
    Truncated Fourier basis in 1D.

    Example
    -------
    >>> basis = TruncatedFourier(n_freq=10)
    >>> model.fit(data, basis)
    >>> U_in_basis, V_in_basis = model.factors
    >>> U, V = model.deconvolve_factors(basis)
    """

    def __init__(self, n_freq):
        self.n_freq = n_freq

    def transform(self, data):

        # Data dimensions.
        if data.ndim == 1:
            data = data[:, None]

        n_obs, n_features = data.shape

        if n_features != 1:
            raise ValueError("Expected 1D data, with shape (n_obs, 1).")

        # Scale each dimension to be on the interval [0, 2 * pi)
        X = data - np.nanmin(data)
        X *= 2 * np.pi / np.nanmax(X)

        # Iterate over each dimension.
        tfmX = [np.ones(X.size)]

        for c in range(1, self.n_freq):
            tfmX.append(np.cos(X @ c))
            tfmX.append(np.sin(X @ c))

        tfmX = np.column_stack(tfmX)

        return tfmX


class TruncatedFourier2D:
    """
    Truncated 2-dimensional Fourier basis.

    Example
    -------
    >>> basis = TruncatedFourier(n_freq=10)
    >>> model.fit(data, basis)
    >>> U_in_basis, V_in_basis = model.factors
    >>> U, V = model.deconvolve_factors(basis)
    """

    def __init__(self, n_freq):
        self.n_freq = n_freq

    def transform(self, data):

        # Data dimensions.
        n_obs, n_features = data.shape

        if n_features != 2:
            raise ValueError("Expected 2D data, with shape (n_obs, 2).")

        # Scale each dimension to be on the interval [0, 2 * pi)
        X = data - np.nanmin(data, axis=0, keepdims=True)
        X *= 2 * np.pi / np.nanmax(X, axis=0, keepdims=True)

        # Iterate over each dimension.
        s = np.array([1.0, -1.0])
        tfmX = []
        prod_iter = itertools.product(
            range(self.n_freq + 1), range(self.n_freq + 1))

        for _c in prod_iter:

            # No need to iterate over signs for constant term.
            if np.sum(_c) == 0:
                tfmX.append(np.ones(n_obs))

            else:
                c = np.array(_c).ravel()
                tfmX.append(np.cos(X @ c))
                tfmX.append(np.sin(X @ c))

                if np.all(c > 0):
                    tfmX.append(np.cos(X @ (s * c)))
                    tfmX.append(np.sin(X @ (s * c)))

        tfmX = np.column_stack(tfmX)

        return tfmX
