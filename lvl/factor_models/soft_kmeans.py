"""
Soft K-means clustering.
"""
import numpy as np
import numba
from scipy.spatial.distance import cdist
from scipy.special import softmax

from numpy.linalg import lstsq

from lvl.exceptions import raise_not_fitted, raise_no_method, raise_no_init
from lvl.utils import get_random_state


class SoftKMeans:
    """
    Specifies Soft K-means clustering model.
    """

    def __init__(
            self, n_components, method="em", init="rand",
            n_restarts=1, seed=None, tol=1e-5, maxiter=100):

        # Model options.
        self.n_components = n_components

        # Optimization parameters.
        self.maxiter = maxiter
        self.seed = seed
        self.tol = tol
        self.n_restarts = n_restarts

        # Model parameters.
        self._factors = None

        # Check that optimization method is recognized.
        METHODS = ("em",)
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

            # Fit model.
            U, Vt = _fit_soft_kmeans(
                np.copy(X), self.n_components, mask,
                self.method, self.init,
                self.maxiter, self.tol, self.seed
            )

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


def _fit_soft_kmeans(
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

    if method == "em":
        return soft_kmeans_em(
            X, rank, mask, init, maxiter, tol, seed)

    else:
        raise NotImplementedError(
            "Did not recognize fitting method.")


def _init_soft_kmeans(X, rank, mask, init, seed):
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


def soft_kmeans_em(X, rank, mask, init, maxiter, tol, seed):
    """
    Fits K-means clustering by expectation maximization.

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

    # Initialize centroids, has shape (rank, n).
    m, n = X.shape
    centroids = _init_soft_kmeans(X, rank, mask, init, seed)
    resp = softmax(-cdist(X, centroids, metric='sqeuclidean'), axis=-1)
    last_resp = np.copy(resp)

    for itr in range(maxiter):

        # Update centroids.
        centroids = lstsq(resp, X, rcond=None)[0]

        # Compute cluster assignments for each datapoint.
        sim = -cdist(X, centroids, metric='sqeuclidean')
        resp = softmax(sim, axis=-1)

        # Update masked elements.
        if mask is not None:
            X[~mask] = (resp @ centroids)[~mask]

        # Check convergence.
        if (np.linalg.norm(resp - last_resp) / np.sqrt(X.shape[0])) < tol:
            break
        else:
            last_resp[:] = resp

    return resp, centroids
