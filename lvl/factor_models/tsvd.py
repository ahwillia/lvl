"""
Truncated Singular Value Decomposition.
"""
import numpy as np
from sklearn.decomposition import TruncatedSVD as _TruncatedSVD

from lvl.exceptions import raise_not_fitted, raise_no_method, raise_no_init
from lvl.utils import get_random_state, rand_orth


class TSVD:
    """
    Specifies Truncated SVD model.
    """

    def __init__(
            self, n_components, method="sklearn", init="rand_orth",
            orthogonalize=True, tol=1e-5, maxiter=100,
            seed=None):

        # Model options.
        self.n_components = n_components
        self.orthogonalize = orthogonalize

        # Optimization parameters.
        self.tol = tol
        self.maxiter = maxiter
        self.seed = seed

        # Model parameters.
        self._factors = None

        # Check that optimization method is recognized.
        METHODS = ("als", "sklearn")
        if method not in METHODS:
            raise_no_method(self, method, METHODS)
        else:
            self.method = method

        # Check that initialization method is recognized.
        INITS = ("rand_orth",)
        if init not in INITS:
            raise_no_init(self, init, INITS)
        else:
            self.init = init

    def fit(self, X, mask=None, overwrite_loss=True):
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
        overwrite_loss : bool
            If True, self.loss_hist is overwritten.
        """

        U, Vt, loss_hist = _fit_tsvd(
            X, self.n_components, mask,
            self.method, self.init, self.tol,
            self.maxiter, self.seed
        )

        # Store factors.
        U = U[:, None] if U.ndim == 1 else U
        Vt = Vt[None, :] if Vt.ndim == 1 else Vt
        self._factors = U, Vt

        # Store loss history.
        if overwrite_loss:
            self.loss_hist = loss_hist

        # If desired, orthogonalize factors.
        if (mask is "als") and self.orthogonalize:
            svd = _TruncatedSVD(
                n_components=self.n_components)
            U = svd.fit_transform(self.predict())
            Vt = svd.components_
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
            raise_not_fitted(self, "factors")


def _fit_tsvd(
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

    if method == "als":
        return tsvd_als(
            X, rank, mask, init, tol, maxiter, seed)

    if method == "sklearn":
        return tsvd_sklearn(
            X, rank, mask, init, tol, maxiter, seed)

    else:
        raise NotImplementedError(
            "Did not recognize fitting method.")


def _init_tsvd(X, rank, mask, init, seed):
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
    if init == "rand_orth":

        # Randomized initialization.
        U = rand_orth(m, rank)
        Vt = rand_orth(rank, n)

        # Determine appropriate scaling.
        e = np.dot(U, Vt) * mask
        alpha = np.sqrt(
            xtx / np.dot(e.ravel(), e.ravel()))

        # Scale randomized initialization.
        U *= alpha
        Vt *= alpha

    else:
        raise NotImplementedError(
            "Did not recognize init method.")

    return U, Vt, xtx


def tsvd_sklearn(X, rank, mask, init, tol, maxiter, seed):
    """
    Fits truncated SVD with sklearn backend, using iterative
    imputation to handle missing data.

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
    U : ndarray
        First factor matrix. Has shape (m, rank).
    Vt : ndarray
        Second factor matrix. Has shape (rank, n).
    loss_hist : ndarray
        Vector holding loss values. Has shape
        (n_iterations,).
    """

    # Simple imputation for initialization.
    if mask is not None:
        X = np.copy(X)
        X[~mask] = np.mean(X[mask])

    # Fit initial model.
    svd = _TruncatedSVD(
        n_components=rank,
        random_state=seed
    )
    U = svd.fit_transform(X)
    Vt = svd.components_

    # We are finished if data is unmasked.
    if mask is None:
        return U, Vt, np.array([])

    # Otherwise we need to continue imputing masked
    # entries until convergence.
    last_pred = (U @ Vt)
    pred = np.empty_like(last_pred)
    for itr in range(maxiter):

        # Impute missing entries.
        X[~mask] = last_pred[~mask]

        # Update model parameters.
        U = svd.fit_transform(X)
        Vt = svd.components_
        pred[:] = (U @ Vt)

        # Check convergence. Due to rotational invariance in the factors,
        # we check on the full model prediction
        upd = np.linalg.norm(last_pred - pred) / np.sqrt(np.prod(X.shape))

        if upd < tol:
            break
        else:
            last_pred[:] = pred

    return U, Vt, np.array([])


def tsvd_als(X, rank, mask, init, tol, maxiter, seed):
    """
    Fits truncated SVD by alternating least squares (ALS).

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
    U : ndarray
        First factor matrix. Has shape (m, rank).
    Vt : ndarray
        Second factor matrix. Has shape (rank, n).
    loss_hist : ndarray
        Vector holding loss values. Has shape
        (n_iterations,).
    """

    U, Vt, xtx = _init_tsvd(X, rank, mask, init, seed)

    m, n = X.shape
    mX = mask * X

    loss_hist = []

    for itr in range(maxiter):

        # Update U.
        U = censored_lstsq(Vt.T, X.T, mask.T).T

        # Update Vt.
        Vt = censored_lstsq(U, X, mask)

        # Record loss.
        if m > n:
            utxv = np.sum(U * np.dot(mX, Vt.T))
        else:
            utxv = np.sum(np.dot(U.T, mX) * Vt)
        utuvtv = np.sum(np.dot(U.T, U) * np.dot(Vt, Vt.T))

        loss_hist.append(xtx - 2 * utxv + utuvtv)

        # Check convergence.
        if (itr > 0) and ((loss_hist[-2] - loss_hist[-1]) < tol):
            break

    return U, Vt, np.array(loss_hist)


def censored_lstsq(A, B, M):
    """
    Solves least squares problem subject to missing data.

    Args
    ----
    A (ndarray) : m x r matrix
    B (ndarray) : m x n matrix
    M (ndarray) : m x n binary matrix (zeros indicate missing values)

    Returns
    -------
    X (ndarray) : r x n matrix that minimizes norm(M*(AX - B))
    """

    # If B is a vector, simply drop out corresponding rows in A
    if B.ndim == 1 or B.shape[1] == 1:
        return np.linalg.leastsq(A[M], B[M])[0]

    # Ensure A is matrix.
    if A.ndim == 1:
        A = A[:, None]

    # If B is a matrix, convert to tensor form.
    #   - right hand side has shape (n, r, 1).
    rhs = np.dot(A.T, M * B).T[:, :, None]
    #   - left hand side has shape (n, r, r).
    lhs = np.matmul(A.T[None, :, :], M.T[:, :, None] * A[None, :, :])
    #   - result has shape (n, r, 1).
    X = np.linalg.solve(lhs, rhs)

    # Squeeze and transpose result to get (r, n) shaped solution.
    return np.squeeze(X).T
