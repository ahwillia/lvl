"""
Nonnegative matrix factorization routines.

References
----------
Gillis, N. (2014). The why and how of nonnegative matrix factorization.
    Regularization, Optimization, Kernels, and Support Vector Machines,
    12(257).
"""
import numpy as np
import numba

from lvl.exceptions import raise_not_fitted, raise_no_method, raise_no_init
from lvl.utils import get_random_state


class NMF:
    """
    Nonnegative Matrix Factorization (NMF) with quadratic loss.
    """

    def __init__(
            self, n_components, method="hals", init="rand",
            tol=1e-5, maxiter=100, seed=None):

        # Model hyperparameters.
        self.n_components = n_components

        # Optimization parameters.
        self.tol = tol
        self.maxiter = maxiter
        self.seed = seed

        # Model parameters.
        self._factors = None

        # Check that optimization method is recognized.
        METHODS = ("hals",)
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
        W, H, self.loss_hist = _fit_nmf(
            X, self.n_components, mask,
            self.method, self.init, self.tol,
            self.maxiter, self.seed
        )
        self._factors = W, H

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
            raise_not_fitted(self, "factors")


def _fit_nmf(
        X, rank, mask, method, init, tol, maxiter, seed):
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
    tol : float
        Convergence tolerance.
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

    if method == "hals":
        return nmf_hals(
            X, rank, mask, init, tol, maxiter, seed)

    else:
        raise NotImplementedError(
            "Did not recognize fitting method.")


def _init_nmf(X, rank, mask, init, seed):
    """
    Dispatches the desired initialization method.

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

    # Data dimensions.
    m, n = X.shape

    # Mask data.
    if mask is not None:
        Xm = mask * X
    else:
        Xm = X

    # Compute norm of masked data.
    xtx = np.dot(Xm.ravel(), Xm.ravel())

    # Seed random number generator.
    rs = get_random_state(seed)

    # Random initialization.
    if init == "rand":

        # Randomized initialization.
        W = rs.rand(m, rank)
        H = rs.rand(rank, n)

        # Determine appropriate scaling.
        if mask is None:
            alpha = np.sqrt(
                xtx / np.sum(np.dot(W.T, W) * np.dot(H, H.T)))
        else:
            e = np.dot(W, H) * mask
            alpha = np.sqrt(
                xtx / np.dot(e.ravel(), e.ravel()))

        # Scale randomized initialization.
        W *= alpha
        H *= alpha

    else:
        raise NotImplementedError(
            "Did not recognize init method.")

    return W, H, xtx


def nmf_hals(X, rank, mask, init, tol, maxiter, seed):
    """
    Fits NMF using Hierarchical Least Squares.

    Parameters
    ----------
    X : ndarray
        Data matrix. Has shape (m, n)
    rank : int
        Number of components.
    mask : ndarray
        Mask for missing data. Has shape (m, n).
    tol : float
        Convergence tolerance.
    maxiter : int
        Number of iterations.
    seed : int or np.random.RandomState
        Seed for random number generator for initialization.

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

    W, H, xtx = _init_nmf(X, rank, mask, init, seed)

    loss_hist = []

    if mask is None:
        update_rule = _hals_update
        inner_iters = 3
        mask_T = None
    else:
        update_rule = _hals_update_with_mask
        inner_iters = 1
        mask_T = mask.T

    for itr in range(maxiter):

        # Update W.
        update_rule(X, W, H, mask, inner_iters)

        # Update H.
        l2 = update_rule(X.T, H.T, W.T, mask_T, inner_iters)

        # Record loss.
        if mask is None:
            loss_hist.append((xtx + l2) / xtx)
        else:
            loss_hist.append(l2 / xtx)

        # Check convergence.
        if (itr > 0) and ((loss_hist[-2] - loss_hist[-1]) < tol):
            break

    return W, H, np.array(loss_hist)


@numba.jit(nopython=True, cache=True)
def _hals_update(X, W, H, mask, n_iters):
    """
    Updates W. Follows notation in:

    Gillis N, Glineur F (2012). Accelerated multiplicative updates
    and hierarchical ALS algorithms for nonnegative matrix
    factorization. Neural computation, 24(4), 1085-1105.
    """

    # Problem dimensions.
    rank = W.shape[1]
    indices = np.arange(rank)

    # Cache gram matrices.
    A = np.dot(X, H.T)
    B = np.dot(H, H.T)

    # Handle special case of rank-1 model.
    if rank == 1:
        W[:] = np.maximum(0.0, A / B)

    # Handle rank > 1 cases.
    else:
        for j in range(n_iters):
            for p in range(rank):
                idx = (indices != p)
                Cp = np.dot(W[:, idx], B[idx][:, p])
                r = (A[:, p] - Cp) / B[p, p]
                W[:, p] = np.maximum(r, 0.0)

    return -2 * np.sum(W * A) + np.sum(np.dot(W.T, W) * B)


@numba.jit(nopython=True, cache=True)
def _hals_update_with_mask(X, W, H, mask, n_iters):
    """
    Updates W.
    """

    rank = W.shape[1]
    indices = np.arange(rank)

    # Handle special case of rank-1 model.
    if rank == 1:
        W[:, 0] = \
            np.dot(mask * X, H[0]) / np.dot(mask * H, H[0])
        mr = mask * (X - np.dot(W, H))
        return np.dot(mr.ravel(), mr.ravel())

    # Handle rank > 1 cases.
    for p in range(rank):
        idx = (indices != p)
        resid = X - np.dot(W[:, idx], H[idx, :])
        r = np.dot(mask * resid, H[p]) / np.dot(mask * H[(p,)], H[p])
        W[:, p] = np.maximum(r, 0.0)

    mr = mask * (resid - np.outer(W[:, -1], H[-1]))
    return np.dot(mr.ravel(), mr.ravel())
