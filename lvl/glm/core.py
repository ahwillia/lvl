import numpy as np
from scipy.special import softmax
from tqdm import trange

from lvl.glm.sgd import sgd_poiss_reg
from lvl.glm.sgd_avg import sgd_avg_poiss_reg
from lvl.glm.svrg import svrg_poiss_reg
from lvl.glm._scipy import scipy_poiss_reg
from lvl.glm.pass_glm import pass_poiss_reg


def poiss_glm(
        X, Y, method="newton-cg", W=None, obs_weights=None, **kwargs):
    """
    Fits a generalized linear model with a Poisson noise model and
    canonical log link function.

    Parameters
    ----------
    X : ndarray
        Independent variables. Has shape (n_obs, n_features).

    Y : ndarray
        Dependent variables. Has shape (n_obs, n_dep_vars).

    method : str
        Specifies fitting method. Valid options are:
        "sgd" (Stochastic Gradient Descent), "newton-cg"
        (Newton's method with conjugate gradient descent, scipy),
        "trust-exact" (Newton's method). Default is "newton-cg".

    W : ndarray or None
        Initial guess for regression weights. If None, an appropriate
        initialization is automatically chosen. Has shape
        (n_dep_vars, n_features).

    obs_weights : ndarray
        Nonnegative weight attached to each observation. This can
        be used, for example, to optimize the expected log-likelihood
        in an expectation-maximization setting. Has shape (n_obs,).

    **kwargs : dict
        Additional keyword arguments

    Returns
    -------
    W : ndarray
        Fitted regression weights. Has shape (n_dep_vars, n_features).

    loss_hist : ndarray
        History of optimization progress. Available for only some solvers.
    """

    if method == "sgd":
        return sgd_poiss_reg(
            X, Y, W=W, obs_weights=obs_weights, **kwargs)

    elif method == "sgd_avg":
        return sgd_avg_poiss_reg(
            X, Y, W=W, obs_weights=obs_weights, **kwargs)

    elif method == "svrg":
        return svrg_poiss_reg(
            X, Y, W=W, obs_weights=obs_weights, **kwargs)

    elif method == "pass":
        return pass_poiss_reg(X, Y, W=W, obs_weights=obs_weights, **kwargs)

    elif method in ("newton-cg", "trust-exact"):
        return scipy_poiss_reg(
            X, Y, method, W=W, obs_weights=obs_weights, **kwargs)

    else:
        raise ValueError("Method not recognized.")


def poiss_glm_mixture(
        n_mixtures, X, Y, glm_method="pass", verbose=True, em_iters=10,
        max_temp=1.0, **kwargs):
    """
    Fits a mixture of GLMs to population data by expectation
    maximization.

    Parameters
    ----------
    n_mixtures : int
        Number of mixture components.

    X : ndarray
        Independent variables. Has shape (n_obs, n_features).

    Y : ndarray
        Dependent variables. Has shape (n_obs, n_dep_vars).

    glm_method : str
        Specifies glm fitting method. See `poiss_glm` function
        for valid options.

    **kwargs : dict
        Additional keyword arguments are passed to the glm
        fitting method.

    Returns
    -------
    z : ndarray
        Probability of each datapoint being assigned
        to each mixture. Has shape (n_mixtures, n_obs).

    W : ndarray
        Fitted regression weights. Has shape (n_mixtures,
        n_dep_vars, n_features).
    """

    n_obs, n_features = X.shape
    assert Y.shape[0] == n_obs
    n_dep_vars = Y.shape[1]

    # Initialize responsibilities for each observation.
    z = softmax(
        np.random.exponential(1.0, size=(n_mixtures, n_obs)), axis=0)
    ll = np.empty_like(z)

    W = np.empty((n_mixtures, n_dep_vars, n_features))
    infos = [None for k in range(n_mixtures)]

    log_like_hist = []

    pbar = trange(em_iters) if verbose else range(em_iters)

    temp = np.concatenate(
        (np.logspace(np.log10(max_temp), 0.0, em_iters // 2),
         np.ones(em_iters - em_iters // 2)))

    for itercount in pbar:

        # M-step
        for k in range(n_mixtures):

            # Warm start for kth mixture.
            _w = W[k] if (itercount > 0) else None

            # Fit regression weights.
            W[k], infos[k] = poiss_glm(
                X, Y, W=_w, obs_weights=z[k], method=glm_method, **kwargs)

        # E-step
        for k in range(n_mixtures):

            # Compute log-likelihoods.
            XW = X @ W[k].T
            ll[k] = np.sum(XW * Y, axis=1) - np.sum(np.exp(XW), axis=1)

        z = softmax(ll / temp[itercount], axis=0)

        log_like_hist.append(np.sum(z * ll) / n_obs)

    return W, z, log_like_hist
