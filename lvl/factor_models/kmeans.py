"""
K-means clustering
"""
import numpy as np
import numba

from lvl.exceptions import raise_not_fitted, raise_no_method, raise_no_init
from lvl.utils import get_random_state


class KMeans:
    """
    Specifies K-means clustering model.
    """

    def __init__(
            self, n_components, method="lloyds", init="rand",
            n_restarts=10, tol=1e-4, seed=None, maxiter=100):

        # Model options.
        self.n_components = n_components

        # Optimization parameters.
        self.maxiter = maxiter
        self.n_restarts = n_restarts
        self.seed = seed
        self.tol = tol

        # Model parameters.
        self._factors = None

        # Check that optimization method is recognized.
        METHODS = ("lloyds",)
        if method not in METHODS:
            raise_no_method(self, method, METHODS)
        else:
            self.method = method

        # Check that initialization method is recognized.
        INITS = ("rand",)
        if init not in INITS:
            raise_no_init(self, init, INITS)
        else:
            self.init = init

    def fit(self, X, mask=None):
        """
        Fits model parameters.

        Parameters
        ----------
        X : ndarray
            Matrix holding data. Has shape (m, n).
        mask : ndarray
            Binary array specifying observed data points
            (where mask == 1) and unobserved data points
            (where mask == 0). Has shape (m, n).
        """

        losses = []
        for itr in range(self.n_restarts):
            assignments, Vt = _fit_kmeans(
                np.copy(X), self.n_components, mask,
                self.method, self.init, self.maxiter,
                self.tol, self.seed
            )

            # Create one-hot representation of cluster assignments.
            U = np.zeros((X.shape[0], self.n_components))
            U[np.arange(X.shape[0]), assignments] = 1.0

            # Compute loss.
            resid = X - np.dot(U, Vt)
            if mask is None:
                loss = np.linalg.norm(resid)
            else:
                loss = np.linalg.norm(mask * resid)

            # Save best factors.
            losses.append(loss)
            if np.argmin(losses) == itr:
                self._factors = U, Vt

    def predict(self):
        return np.dot(*self.factors)

    def score(self, X, mask=None):
        """
        Computes goodness-of-fit score.

        Parameters
        ----------
        X : ndarray
            Matrix holding data. Has shape (m, n).
        mask : ndarray
            Binary array specifying observed data points
            (where mask == 1) and unobserved data points
            (where mask == 0). Has shape (m, n).

        Returns
        -------
        model_score : float
            One minus the norm of model residuals divided
            by the norm of the raw data. A score of zero
            corresponds to no fit. A score of one corresponds
            to a perfect fit.
        """

        # Compute low-rank prediction.
        pred = self.predict()

        # Check dimensions.
        if pred.shape != X.shape:
            raise ValueError(
                "Model was fit to data with shape {}, "
                "but asked to score data with shape "
                "{}. Dimensions must match"
                ".".format(pred.shape, X.shape))

        # Default to fully observed data.
        if mask is None:
            mask = np.ones_like(X)

        # Compute performance score.
        resid_norm = np.linalg.norm(mask * (pred - X))
        data_norm = np.linalg.norm(mask * X)
        return 1.0 - resid_norm / data_norm

    @property
    def factors(self):
        self._assert_fitted()
        return self._factors

    def _assert_fitted(self):
        if self._factors is None:
            raise_not_fitted(
                type(self).__name__, "factors")


def _fit_kmeans(
        X, rank, mask, method, init, maxiter, tol, seed):
    """
    Dispatches the desired optimization method.

    Parameters
    ----------
    X : ndarray
        Data matrix. Has shape (m, n)
    rank : int
        Number of components.
    mask : ndarray
        Mask for missing data. Has shape (m, n).
    init : str
        Specifies initialization method.
    maxiter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance.
    seed : int or numpy.random.RandomState
        Seeds random number generator.

    Returns
    -------
    W : ndarray
        First factor matrix. Has shape (m, rank).
    H : ndarray
        Second factor matrix. Has shape (rank, n).
    loss_hist : ndarray
        Vector holding loss values. Has shape
        (n_iterations,).
    """

    if method == "lloyds":
        return kmeans_lloyds(
            X, rank, mask, init, maxiter, tol, seed)

    else:
        raise NotImplementedError(
            "Did not recognize fitting method.")


def _init_kmeans(X, rank, mask, init, seed):
    """
    Dispatches the desired initialization method.

    Parameters
    ----------
    X : ndarray
        Data matrix. Has shape (m, n)
    rank : int
        Number of cluster centroids.
    mask : ndarray
        Mask for missing data. Has shape (m, n).
    init : str
        Specifies initialization method.
    seed : int or numpy.random.RandomState
        Seeds random number generator.

    Returns
    -------
    W : ndarray
        First factor matrix. Has shape (m, rank).
    H : ndarray
        Second factor matrix. Has shape (rank, n).
    xtx : float
        Squared Frobenius norm of X. This is later
        used to scale the model loss.
    """

    # Seed random number generator.
    rs = get_random_state(seed)

    # Random initialization.
    if init == "rand":
        idx = rs.choice(X.shape[0], size=rank, replace=False)
        centroids = X[idx]

    else:
        raise NotImplementedError(
            "Did not recognize init method.")

    return centroids


def kmeans_lloyds(X, rank, mask, init, maxiter, tol, seed):
    """
    Fits K-means clustering by standard method (Lloyd's algorithm).

    Parameters
    ----------
    X : ndarray
        Data matrix. Has shape (m, n)
    rank : int
        Number of cluster centroids.
    mask : ndarray
        Mask for missing data. Has shape (m, n).
    maxiter : int
        Number of iterations.
    tol : float
        Convergence tolerance.
    seed : int or np.random.RandomState
        Seed for random number generator for initialization.

    Returns
    -------
    assignments : ndarray
        Vector holding cluster assignments of each datapoint.
        Has shape (m,). Values are integers on the interval
        [0, n_clusters).
    centroids : ndarray
        Matrix holding estimates of cluster centroids. Has shape
        (n_clusters, n).
    """

    # Simple data imputation for initialization.
    if mask is not None:
        X = np.copy(X)
        X[~mask] = np.nanmean(X[mask])
        row_idx, col_idx = np.where(~mask)  # missing indices
    else:
        # no missing indices.
        row_idx = np.array([], dtype='int')
        col_idx = np.array([], dtype='int')

    # Initialize centroids and allocate space for assignments.
    m, n = X.shape
    centroids = _init_kmeans(X, rank, mask, init, seed)
    assignments = np.empty(m, dtype='int')
    cluster_sizes = np.empty(m, dtype='int')
    last_assignments = np.full(m, -1)
    last_centroids = np.empty_like(centroids)

    for itr in range(maxiter):

        # Compute cluster assignments for each datapoint.
        _update_lloyds(
            X, centroids, assignments, cluster_sizes, row_idx, col_idx)

        # Check convergence.
        sgn_cvg = np.all(last_assignments == assignments)
        cent_cvg = (
            np.linalg.norm(centroids - last_centroids) /
            np.linalg.norm(centroids)) < tol
        if sgn_cvg and cent_cvg:
            break
        else:
            last_assignments[:] = assignments
            last_centroids[:] = centroids

    return assignments, centroids


@numba.jit(nopython=True, cache=True)
def _update_lloyds(
        X, centroids, assignments, cluster_sizes, row_idx, col_idx):
    """Performs one step of Lloyd's algorithm."""

    I, J = X.shape
    K = centroids.shape[0]

    # == UPDATE ASSIGNMENTS == #
    cluster_sizes.fill(0)
    for i in range(I):

        # Find nearest centroid.
        best_dist = np.inf
        for k in range(K):
            dist = 0.0

            for j in range(J):
                dist += (X[i, j] - centroids[k, j]) ** 2
                if dist > best_dist:
                    break

            if dist < best_dist:
                assignments[i] = k
                best_dist = dist

        # Update cluster sizes.
        cluster_sizes[assignments[i]] += 1

    # == UPDATE CENTROIDS == #
    centroids.fill(0.0)
    for i in range(I):
        centroids[assignments[i]] += X[i]
    for k in range(K):
        centroids[k] /= cluster_sizes[k]

    # == UPDATE MISSING ELEMENTS == #
    for i, j in zip(row_idx, col_idx):
        X[i, j] = centroids[assignments[i]][j]
