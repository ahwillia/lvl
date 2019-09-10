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

from lvl.nnls import nnls
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
        if n_components <= 0:
            raise ValueError("Expected n_components to be an integer >= 1.")

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

    def bicv_extend(self, B, C):
        """
        Extends model fit for bi-cross-validation.
        """

        # Get previously fit model factors.
        W, H = self.factors

        # Check inputs.
        assert B.shape[0] == W.shape[0]
        assert C.shape[1] == H.shape[1]

        # Extend factors via nonnegative least squares, fit
        # with coordinate descent (cd).
        H_ = nnls(W, B, method="cd", tol=self.tol)
        W_ = nnls(H.T, C.T, method="cd", tol=self.tol).T
        self._factors = (
            np.row_stack((W, W_)),
            np.column_stack((H, H_))
        )

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
        Binary array specifying observed data points
        (where mask == 1) and unobserved data points
        (where mask == 0). Has shape (m, n).
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

    m, n = X.shape
    loss_hist = []

    if mask is not None:
        X = np.copy(X)
        X[~mask] = np.mean(X[mask])
        Xpred = np.empty((m, n))

    # Allocate space for intermediate computations.
    XHt = X @ H.T
    HHt = H @ H.T
    WtX = np.empty((rank, n))
    WtW = np.empty((rank, rank))

    for itr in range(maxiter):

        # Update W.
        _hals_update(W, XHt, HHt, 3)

        if mask is not None:
            np.dot(W, H, out=Xpred)
            X[~mask] = Xpred[~mask]

        np.dot(W.T, X, out=WtX)
        np.dot(W.T, W, out=WtW)

        # Update H.
        _hals_update(H.T, WtX.T, WtW, 3)

        if mask is not None:
            np.dot(W, H, out=Xpred)
            X[~mask] = Xpred[~mask]

        np.dot(X, H.T, out=XHt)
        np.dot(H, H.T, out=HHt)

        # Record loss.
        if mask is None:
            loss_hist.append(
                (xtx - 2 * np.sum(W * XHt) + np.sum(WtW * HHt)) / xtx
            )
        else:
            resid = X - Xpred
            num = np.dot(resid.ravel(), resid.ravel())
            loss_hist.append(num / xtx)

        # Check convergence.
        if (itr > 0) and ((loss_hist[-2] - loss_hist[-1]) < tol):
            break

    return W, H, np.array(loss_hist)


@numba.jit(nopython=True, cache=True)
def _hals_update(W, XHt, HHt, n_iters):
    """
    Updates W. Follows notation in:

    Gillis N, Glineur F (2012). Accelerated multiplicative updates
    and hierarchical ALS algorithms for nonnegative matrix
    factorization. Neural computation, 24(4), 1085-1105.
    """

    # Problem dimensions.
    rank = W.shape[1]
    indices = np.arange(rank)

    # Handle special case of rank-1 model.
    if rank == 1:
        W[:] = np.maximum(0.0, XHt / HHt)

    # Handle rank > 1 cases.
    else:
        for j in range(n_iters):
            for p in range(rank):
                idx = (indices != p)
                Cp = np.dot(W[:, idx], HHt[idx][:, p])
                r = (XHt[:, p] - Cp) / HHt[p, p]
                W[:, p] = np.maximum(r, 0.0)
