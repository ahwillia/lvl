import numpy as np
from sklearn.decomposition import TruncatedSVD as _TruncatedSVD

from lvl.exceptions import raise_not_fitted, raise_no_method, raise_no_init
from lvl.utils import get_random_state, rand_orth

from tqdm import trange
import numba


class PoissonMF:
    """
    Specifies Matrix Factorization with a Poisson loss
    function.
    """

    def __init__(
            self, n_components, method="cd",
            tol=1e-5, maxiter=100, seed=None):

        # Model options.
        self.n_components = n_components

        # Optimization parameters.
        self.tol = tol
        self.maxiter = maxiter
        self.seed = seed

        # Model parameters.
        self._factors = None

        # Check that optimization method is recognized.
        METHODS = ("cd",)
        if method not in METHODS:
            raise_no_method(self, method, METHODS)
        else:
            self.method = method

    def fit(self, X, mask=None, Vbasis=None):
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
        U, Vt, self.loss_hist = _fit_poiss_mf(
            self.method, X, self.n_components,
            mask, Vbasis, self.tol, self.maxiter, self.seed
        )
        self._factors = U, Vt

    def predict(self):
        return np.exp(np.dot(*self.factors))

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
        """
        raise NotImplementedError

    @property
    def factors(self):
        self._assert_fitted()
        return self._factors

    def _assert_fitted(self):
        if self._factors is None:
            raise_not_fitted(self, "factors")


def _fit_poiss_mf(method, *args):
    """Dispatches desired optimization method."""
    if method == "cd":
        return poisson_mf_cd(*args)
    else:
        raise ValueError("Did not recognize method.")


# ====================================================== #
# ================ OPTIMIZATION METHODS ================ #
# ====================================================== #

def poisson_mf_cd(
        X, rank, mask, Vbasis, tol, maxiter, seed):
    """
    Parameters
    ----------
    X : ndarray
        Matrix holding inputs data. Has shape (m, n).
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

    X = np.asarray(X, dtype='float')

    # Initialize parameters.
    loss_hist = []
    m, n = X.shape
    rs = get_random_state(seed)

    # Account for masked entries.
    if mask is not None:
        X = np.copy(X)
        X[~mask] = np.mean(X[mask])
        Xpred = np.empty((m, n))

    # Initialize parameters.
    U = rs.uniform(-1, 1, size=(m, rank))
    if Vbasis is None:
        Vt = rs.uniform(-1, 1, size=(rank, n))
    else:
        Vt = rs.uniform(-1, 1, size=(rank, Vbasis.shape[0]))

    # Convergence check on parameters.
    Ulast = np.empty_like(U)
    Vlast = np.empty_like(Vt)

    # Main optimization loop.
    for itr in range(maxiter):

        # Update U.
        if Vbasis is None:
            _poiss_cd_update(X, U, Vt, mask)
        else:
            _poiss_cd_update(X, U, Vt @ Vbasis, mask)

        # Update V.
        if Vbasis is None:
            ls = _poiss_cd_update(X.T, Vt.T, U.T, mask_T)
        else:
            ls = _poiss_cd_update_with_basis(X.T, Vt.T, U.T, mask_T, Vbasis)

        # Update masked elements.
        if mask is not None:
            np.dot(U, Vt, out=Xpred)
            X[~mask] = Xpred[~mask]

        # Store loss.
        loss_hist.append(ls / X.size)

        # Check convergence.
        U_upd = np.linalg.norm(Ulast - U) / np.linalg.norm(U)
        V_upd = np.linalg.norm(Vlast - V) / np.linalg.norm(Vt)
        if (itr > 0) and (U_upd < tol) and (V_upd < tol):
            break

        # Make copies of previous parameters.
        np.copyto(Ulast, U)
        np.copyto(Vlast, Vt)

    return U, Vt, np.array(loss_hist)


@numba.jit(nopython=True, cache=True)
def _poiss_cd_update(Y, U, Vt, mask):
    """Updates U."""

    m, n = Y.shape
    rank = U.shape[1]
    U_last = np.empty(U.shape[0])

    eUV = np.exp(np.dot(U, Vt))
    new_loss = np.sum(eUV) - np.sum(np.dot(U.T, Y) * Vt)

    for r in np.random.permutation(rank):

        # Store current loss.
        last_loss = new_loss

        # Compute search direction, as inverse Hessian times gradient.
        # (the Hessian is diagonal, so the inverse is simple division).
        search_dir = np.dot(eUV - Y, Vt[r]) / np.dot(eUV, Vt[r] * Vt[r])

        # Search for update with backtracking.
        new_loss = np.inf
        U_last[:] = U[:, r]
        ss = 1.0

        while new_loss > last_loss:
            U[:, r] = U_last - ss * search_dir
            eUV = np.exp(np.dot(U, Vt))
            new_loss = np.sum(eUV) - np.sum(np.dot(U.T, Y) * Vt)
            ss *= 0.5
            if ss < 1e-4:
                U[:, r] = U_last
                break

    return new_loss


@numba.jit(nopython=True, cache=True)
def _poiss_cd_update_with_basis(Y, U, Vt, mask, Bt):
    """Updates U."""

    m, n = Y.shape
    rank = U.shape[1]
    U_last = np.empty(U.shape[0])

    BtY = Bt @ Y
    eUV = np.exp(Bt.T @ U @ Vt)
    new_loss = np.sum(eUV) - np.sum((U.T @ BtY) * Vt)

    for r in np.random.permutation(rank):

        # Store current loss.
        last_loss = new_loss

        # Compute search direction, as inverse Hessian times gradient.
        # (the Hessian is diagonal, so the inverse is simple division).
        grad = Bt @ ((eUV - Y) @ Vt[r])
        hess = (Bt * Bt) @ (eUV - Y) @ (Vt[r] * Vt[r])
        search_dir = grad / hess

        # Search for update with backtracking.
        new_loss = np.inf
        U_last[:] = U[:, r]
        ss = 1.0

        while new_loss > last_loss:
            U[:, r] = U_last - ss * search_dir
            eUV = np.exp((Bt.T @ U) @ Vt)
            new_loss = np.sum(eUV) - np.sum((U.T @ BtY) * Vt)
            ss *= 0.5
            if ss < 1e-4:
                U[:, r] = U_last
                break

    return new_loss
