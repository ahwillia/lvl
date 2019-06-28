import numpy as np
from sklearn.decomposition import TruncatedSVD as _TruncatedSVD

from lvl.exceptions import raise_not_fitted, raise_no_method, raise_no_init
from lvl.utils import get_random_state, rand_orth


class GMF:
    """
    Specifies Generalized Matrix Factorization (GMF) model,
    which extends generalized linear modeling principles to
    unsupervised matrix factorization.
    """

    def __init__(
            self, n_components, loss, method="cd",
            init="rand", tol=1e-5, maxiter=100, seed=None):

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
        raise NotImplementedError

    def predict(self):
        return self._invlink(np.dot(*self.factors))

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
