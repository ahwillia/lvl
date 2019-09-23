"""
Soft K-means clustering with smooth drift term.

Note: This is a prototype / experimental feature.
"""
import numpy as np
import numba
from scipy.spatial.distance import cdist

from numpy.linalg import lstsq

from lvl.exceptions import raise_not_fitted, raise_no_method, raise_no_init
from lvl.utils import get_random_state, softmax
from lvl.factor_models.soft_kmeans import soft_kmeans_em

from tqdm import trange


class _NonstationarySoftKMeans:

    def __init__(
            self, n_components, init="rand", smoothness=1.0,
            temperature=1.0, n_restarts=1, seed=None, tol=1e-5,
            maxiter=100):

        # Model options.
        self.n_components = n_components
        self.smoothness = smoothness
        self.temperature = temperature

        # Optimization parameters.
        self.maxiter = maxiter
        self.seed = seed
        self.tol = tol
        self.n_restarts = n_restarts

        # Model parameters.
        self._factors = None

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
            U, Vt = nonstat_soft_kmeans(
                np.copy(X), self.n_components,
                self.smoothness, self.temperature, mask,
                self.init, self.maxiter, self.tol,
                self.seed
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
        return _predict(self._factors[0], self._factors[1])

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


def nonstat_soft_kmeans(X, rank, smoothness, temperature, mask, init, maxiter, tol, seed):
    """
    Fits K-means clustering by standard method (Lloyd's algorithm).

    Parameters
    ----------
    X : ndarray
        Data matrix. Has shape (m, n)
    rank : int
        Number of cluster centroids.
    smoothness : float
        Strength of smoothness regularization.
    temperature : float
        Parameter controlling sparsity of k-means.
    mask : ndarray
        Mask for missing data. Has shape (m, n).
    init : str
        Initialization method.
    maxiter : int
        Number of iterations.
    tol : float
        Convergence tolerance.
    seed : int or np.random.RandomState
        Seed for random number generator for initialization.

    Returns
    -------
    responsibilities : ndarray
        Probability vectors holding soft cluster assignment for
        each datapoint. Has shape (m, rank).
    centroid_tensor : ndarray
        Tensor holding moving estimates of cluster centroids. Has
        shape (m, rank, n).
    """

    # Simple data imputation for initialization.
    if mask is not None:
        X = np.copy(X)
        X[~mask] = np.nanmean(X[mask])

    # Initialize with stationary soft k-means.
    resp, centroids = soft_kmeans_em(
        X, rank, mask, init, maxiter, tol, seed)

    # Initialize centroid tensor, has shape (m, rank, n).
    centroid_tens_last = np.tile(centroids[None, :, :], (X.shape[0], 1, 1))
    centroid_tens = np.copy(centroid_tens_last)

    # Compute responsibilities.
    resp = softmax(_compute_resp(centroid_tens, X) / temperature, axis=-1)

    for itr in range(maxiter):

        # Update centroids.
        _centroid_update(
            resp, centroid_tens, X, smoothness)

        # Update responsibilities.
        resp = softmax(_compute_resp(centroid_tens, X) / temperature, axis=-1)

        # Update masked elements.
        if mask is not None:
            X[~mask] = _predict(resp, centroids)[~mask]

        # # Check convergence.
        # update_norm = np.linalg.norm(centroid_tens - centroid_tens_last)
        # if (update_norm / np.sqrt(centroid_tens.size)) < tol:
        #     break
        # else:
        #     centroid_tens_last[:] = centroid_tens

    return resp, centroid_tens


@numba.jit(nopython=True)
def _compute_resp(centroid_tens, X):
    m, rank, n = centroid_tens.shape
    resp = np.empty((m, rank))
    for i in range(m):
        for r in range(rank):
            resp[i, r] = 0.0
            for j in range(n):
                resp[i, r] += (X[i, j] - centroid_tens[i, r, j]) ** 2
    return resp


@numba.jit(nopython=True, cache=True)
def _centroid_update(resp, ctens, X, lam):

    m, rank, n = ctens.shape

    for i in np.random.permutation(m):

        rhs = resp[i].reshape(-1, 1) * X[i].reshape(1, -1)

        if i == 0:
            rhs += lam * (2 * ctens[i + 1] - ctens[i + 2])
            z = 1.0 / lam

        elif i == 1:
            rhs += lam * (
                2 * ctens[i - 1] + 4 * ctens[i + 1] - ctens[i + 2])
            z = 1.0 / (lam * 5.0)

        elif i == (m - 2):
            rhs += lam * (
                4 * ctens[i - 1] - ctens[i - 2] + 2 * ctens[i + 1])
            z = 1.0 / (lam * 5.0)

        elif i == (m - 1):
            rhs += lam * (2 * ctens[i - 1] - ctens[i - 2])
            z = 1.0 / lam

        else:
            rhs += lam * (4 * ctens[i - 1] - ctens[i - 2] + 4 * ctens[i + 1] - ctens[i + 2])
            z = 1.0 / (lam * 6.0)

        Az = resp[i] * z
        inv = (Az.reshape(-1, 1) * Az.reshape(1, -1)) / (-1.0 - np.dot(Az, resp[i]))
        for r in range(rank):
            inv[r, r] += z

        # Sanity check for Sherman–Morrison formula
        # inv2 = (
        #   np.linalg.inv(np.diag(np.ones(rank) / z) +
        #   resp[i][:, None] * resp[i][None, :]))
        # assert inv2 == inv
        # import ipdb; ipdb.set_trace()

        ctens[i] = np.dot(inv, rhs)


@numba.jit(nopython=True, parallel=True)
def _predict(resp, ctens):

    m, rank, n = ctens.shape
    out = np.empty((m, n))

    for i in numba.prange(m):
        for j in range(n):
            out[i, j] = 0.0
            for r in range(rank):
                out[i, j] += resp[i, r] * ctens[i, r, j]

    return out
