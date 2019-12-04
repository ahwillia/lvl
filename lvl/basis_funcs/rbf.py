import numpy as np
import numba


class GaussianRBF:
    """
    Regular spaced grid of Gaussian Radial Basis Functions.
    """

    def __init__(self, grid_shape):

        # Check grid_shape inputs.
        self.grid_shape = np.asarray(grid_shape)
        if not np.issubdtype(self.grid_shape.dtype, np.integer):
            raise ValueError("Expected integer grid_shape.")

        if self.grid_shape.ndim != 1:
            raise ValueError("Expected grid_shape to be 1D array-like.")

        if self.grid_shape.min() < 2:
            raise ValueError(
                "At least 2 basis functions are needed along each dimension.")

        # Number of radial basis functions.
        self.n_basis_funcs = np.prod(self.grid_shape)
        self.ndim = self.grid_shape.size

        # Compute the centers of the RBFs (n_basis_funcs, ndim)
        grids = [np.linspace(0, 1, s) for s in self.grid_shape]
        self.centers = np.column_stack(
            [g.ravel() for g in np.meshgrid(*grids)])

    def transform(self, data):

        # Data dimensions.
        if data.ndim == 1:
            data = data[:, None]

        n_obs, n_features = data.shape

        if n_features != len(self.grid_shape):
            raise ValueError(
                "Data dimension does not match grid_shape. "
                "Expected data.shape[1] == len(grid_shape).")

        # Scale each dimension to be on the interval [0, 1)
        X = data - np.nanmin(data)
        X /= np.nanmax(X)

        # Compute difference between each datapoint.
        tfmX = _fast_rbf_dist(X, self.centers, self.grid_shape)

        return tfmX



@numba.jit(nopython=True)
def _fast_rbf_dist(X, cent, grid_shape):

    n_obs, ndim = X.shape
    n_basis_funcs = cent.shape[0]

    tfmX = np.empty((n_obs, n_basis_funcs))

    for i in range(n_obs):
        for j in range(n_basis_funcs):

            dist = 0.0
            for k in range(ndim):
                resid = X[i, k] - cent[j, k]
                dist += resid * resid * (grid_shape[k] ** 2)

            tfmX[i, j] = np.exp(-dist)

    return tfmX
