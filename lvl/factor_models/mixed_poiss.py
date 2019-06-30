import numpy as np
from sklearn.decomposition import TruncatedSVD as _TruncatedSVD

from lvl.factor_models.poiss_mf import _poiss_cd_update
from lvl.factor_models.poiss_mf import _poiss_cd_update_with_mask

from lvl.exceptions import raise_not_fitted, raise_no_method, raise_no_init
from lvl.utils import get_random_state, rand_orth


class MixedPoissonMF:
    """
    Predicts data using a generalized linear model plus a
    low-rank factor model.
    """

    def __init__(
            self, n_components, loss, method="cd",
            init="randn", tol=1e-5, maxiter=100, seed=None):

        # Model options.
        self.n_components = n_components
        self.loss = loss

        LOSSES = ("poisson",)
        if loss not in LOSSES:
            raise ValueError(
                "Did not recognize 'loss' option. "
                "Saw '{}'. Expected one of {}."
                "".format(loss, LOSSES))

        # Determine link and inverse link functions.
        INVLINKS = {
            "poisson": np.exp,
        }
        LINKS = {
            "poisson": np.log
        }
        self._invlink = INVLINKS[loss]
        self._link = LINKS[loss]

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

        # Check that initialization method is recognized.
        INITS = ("randn",)
        if init not in INITS:
            raise_no_init(self, init, INITS)
        else:
            self.init = init

    def fit(self, inputs, Y, mask=None):
        """
        Fits model parameters.

        Parameters
        ----------
        inputs : ndarray
            Matrix holding inputs data. Has shape
            (n_inputs, n_obs).
        Y : ndarray
            Matrix holding data. Has shape
            (n_features, n_obs).
        mask : ndarray
            Binary array specifying observed data points
            (where mask == 1) and unobserved data points
            (where mask == 0). Has shape
            (n_features, n_obs).
        """
        U, Vt, weights, self.loss_hist = _fit_mx_glm(
            inputs, Y, self.n_components, mask,
            self.method, self.init, self.tol,
            self.maxiter, self.seed
        )
        self._factors = U, Vt

    def predict(self, inputs):
        return self._invlink(
            np.dot(self.glm_weights, inputs) +
            np.dot(*self.factors))

    def glm_predict(self, inputs):
        return self._invlink(
            np.dot(self.glm_weights, inputs))

    def score(self, inputs, Y, mask=None):
        """
        Computes goodness-of-fit score.

        Parameters
        ----------
        inputs : ndarray
            Matrix holding inputs data. Has shape
            (n_inputs, n_obs).
        Y : ndarray
            Matrix holding data. Has shape
            (n_features, n_obs).
        mask : ndarray
            Binary array specifying observed data points
            (where mask == 1) and unobserved data points
            (where mask == 0). Has shape
            (n_features, n_obs).

        Returns
        -------
        model_score : float
        """
        raise NotImplementedError

    @property
    def factors(self):
        self._assert_fitted()
        return self._factors

    @property
    def glm_weights(self):
        self._assert_fitted()
        return self._glm_weights

    def _assert_fitted(self):
        if self._factors is None:
            raise_not_fitted(self, "factors")


def _fit_mixed_poiss(method, *args):

    if method == "cd":
        return mixed_poiss_cd(*args)
    else:
        raise ValueError("Did not recognize method.")


def mixed_poiss_cd(
        X, Y, rank, mask, tol, maxiter, seed):
    """
    Parameters
    ----------
    X : ndarray
        Matrix holding inputs data. Has shape
        (n_inputs, n_obs).
    Y : ndarray
        Matrix holding data. Has shape
        (n_features, n_obs).
    mask : ndarray
        Binary array specifying observed data points
        (where mask == 1) and unobserved data points
        (where mask == 0). Has shape
        (n_features, n_obs).
    """

    assert X.shape[1] == Y.shape[1]
    n_in, n_obs = X.shape
    n_features, n_obs = Y.shape

    # Initialize parameters.
    rs = get_random_state(seed)
    U = rs.uniform(-1, 1, size=(n_features, n_in + rank))
    Vt = rs.uniform(-1, 1, size=(n_in + rank, n_obs))
    Vt[:n_in] = X

    if mask is None:
        update_rule = _poiss_cd_update
        mask_T = None
    else:
        update_rule = _poiss_cd_update_with_mask
        mask_T = mask.T

    for itr in range(maxiter):

        # Update U.
        update_rule(Y, U, Vt, mask, inner_iters)

        # Update rows of V without over-writing inputs (X).
        ut = U[:, n_in:].T
        v = Vt[n_in:]
        ls = update_rule(Y.T, v, ut, mask_T, inner_iters)

        # Check convergence.
        loss_hist.append(ls)
        if itr > 0 and ((loss_hist[-2] - loss_hist[-1]) < tol):
            break

    return W, H, np.array(loss_hist)
