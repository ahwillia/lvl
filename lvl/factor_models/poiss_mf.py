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
        U, Vt, self.loss_hist = _fit_poiss_mf(
            self.method, X, self.n_components,
            mask, self.tol, self.maxiter, self.seed
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
        X, rank, mask, tol, maxiter, seed):
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
    m, n = X.shape
    rs = get_random_state(seed)
    U = rs.uniform(-1, 1, size=(m, rank))
    Vt = rs.uniform(-1, 1, size=(rank, n))
    loss_hist = []

    if mask is None:
        update_rule = _poiss_cd_update
        mask_T = None
    else:
        update_rule = _poiss_cd_update_with_mask
        mask_T = mask.T

    for itr in range(maxiter):

        # Update U.
        ls = update_rule(X, U, Vt, mask)

        # Update V.
        ls = update_rule(X.T, Vt.T, U.T, mask_T)

        # Check convergence.
        loss_hist.append(ls)
        if itr > 0 and ((loss_hist[-2] - loss_hist[-1]) < tol):
            break

    return U, Vt, np.array(loss_hist)


@numba.jit(nopython=True)
def _poiss_cd_update(Y, U, Vt, mask):
    """Updates U."""

    m, n = Y.shape
    rank = U.shape[1]

    eUV = np.exp(np.dot(U, Vt))
    new_loss = np.sum(eUV) - np.sum(np.dot(U.T, Y) * Vt)

    for r in np.random.permutation(rank):

        # Store current loss.
        last_loss = new_loss

        # Compute search direction. Gradient divided by Hessian
        # (note that the Hessian is diagonal).
        search_dir = np.dot(eUV - Y, Vt[r]) / np.dot(eUV, Vt[r] * Vt[r])

        # Search for update with backtracking.
        new_loss = np.inf
        ss = 1.0

        while new_loss > last_loss:
            U[:, r] = U[:, r] - ss * search_dir
            eUV = np.exp(np.dot(U, Vt))
            new_loss = np.sum(eUV) - np.sum(np.dot(U.T, Y) * Vt)
            ss *= 0.5
            if ss < 1e-4:
                break

    return new_loss


def _poiss_cd_update_with_mask(*args):
    raise NotImplementedError
