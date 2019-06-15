"""
K-means clustering
"""
import numpy as np
import numba

from ..exceptions import raise_not_fitted, raise_no_method, raise_no_init
from ..utils import get_random_state


class KMeans:
    """
    Specifies K-means clustering model.
    """

    def __init__(
            self, n_components, method="lloyds", init="rand",
            seed=None, maxiter=100):

        # Model options.
        self.n_components = n_components

        # Optimization parameters.
        self.maxiter = maxiter
        self.seed = seed

        # Model parameters.
        self._factors = None

        # Check that optimization method is recognized.
        METHODS = ("lloyds",)
        if method not in METHODS:
            raise_no_method("KMeans", method, METHODS)
        else:
            self.method = method

        # Check that initialization method is recognized.
        INITS = ("rand",)
        if init not in INITS:
            raise_no_init("KMeans", init, INITS)
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
        assignments, centroids = _fit_kmeans(
            np.copy(X), self.n_components, mask,
            self.method, self.init,
            self.maxiter, self.seed
        )

        # Create one-hot representation of cluster assignments.
        U = np.zeros((X.shape[0], self.n_components))
        U[np.arange(X.shape[0]), assignments] = 1.0

        self._factors = U, centroids

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
            raise_not_fitted("NMF", "factors")


def _fit_kmeans(
        X, rank, mask, method, init, maxiter, seed):
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
            X, rank, mask, init, maxiter, seed)

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


def kmeans_lloyds(X, rank, mask, init, maxiter, seed):
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
        X[~mask] = np.nanmean(X[mask])

    # Initialize centroids and allocate space for assignments.
    m, n = X.shape
    centroids = _init_kmeans(X, rank, mask, init, seed)
    assignments = np.empty(m, dtype=int)
    last_assignments = np.full(m, -1)

    # Set masked elements to NaN.
    if mask is not None:
        X[~mask] = np.nan

    for itr in range(maxiter):

        # Compute cluster assignments for each datapoint.
        _assign_clusters(X, centroids, assignments)

        # Update centroids.
        for k in range(rank):
            centroids[k] = np.nanmean(X[assignments == k], axis=0)

        # Check convergence.
        if np.all(last_assignments == assignments):
            break
        else:
            last_assignments[:] = assignments

    return assignments, centroids


@numba.jit(nopython=True, parallel=True)
def _assign_clusters(X, centroids, assignments):
    """
    Assign each datapoint to closest cluster, ignoring nans.

    Parameters
    ----------
    X : ndarray
        Data matrix. Has shape (m, n). NaN values are considered
        to be missing data.
    centroids : ndarray
        Matrix holding estimates of cluster centroids. Has shape
        (n_clusters, n).
    assignments : ndarray
        Vector holding cluster assignments of each datapoint.
        Has shape (m,). Values are integers on the interval
        [0, n_clusters).
    """

    I, J = X.shape
    K = centroids.shape[0]

    for i in range(I):

        best_dist = np.inf

        for k in range(K):
            dist = 0.0

            for j in range(J):
                if not np.isnan(X[i, j]):
                    dist += (X[i, j] - centroids[k, j]) ** 2

            # if not np.isfinite(dist):
            #     import ipdb
            #     ipdb.set_trace()

            if dist < best_dist:
                assignments[i] = k
                best_dist = dist
