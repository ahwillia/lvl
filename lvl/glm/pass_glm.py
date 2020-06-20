"""
Fits Poisson GLM with Polynomial Approximated Sufficient
Statistics (PASS) framework. Unlike the references below we
fit the polynomial approximation to the inverse-link function
iteratively, along with the regression weights using
least-squares.

Convergence may not be guarunteed by this scheme but it seems
to work very well in practice and it has the advantage of
automatically estimating the interval for estimating the
nonlinearity.


References
----------
Zoltowski & Pillow (2018). Scaling the Poisson GLM to massive
neural datasets through polynomial approximations. NeurIPS.

Huggins et al. (2017). PASS-GLM: polynomial approximate sufficient
statistics for scalable Bayesian GLM inference. NeurIPS.
"""
import numpy as np
from scipy.special import gammaln
from scipy.linalg import cho_factor, cho_solve
from collections import namedtuple

Info = namedtuple(
    "Info",
    ("loss_hist", "approx_hist", "poly_coef")
)


def pass_poiss_reg(
        X, Y, W=None, obs_weights=None, tol=1e-3, suff_stats=None,
        store_trace=False, max_iter=10, l2_reg=0.0):

    # Ensure Y is 2d.
    if Y.ndim == 1:
        Y = Y[:, None]

    # By default, unweighted observations.
    if obs_weights is None:
        obs_weights = np.ones(Y.shape[0])

    # Compute sufficient statistics.
    if suff_stats is None:
        s_X = X.T @ obs_weights
        Xt_Y = X.T @ (Y * obs_weights[:, None])
        Xt_X = X.T @ (X * obs_weights[:, None])
        Xt_X[np.diag_indices_from(Xt_X)] += l2_reg
        cho_Xt_X = cho_factor(Xt_X)

    else:
        s_X, Xt_Y, cho_Xt_X = suff_stats

    # Handle 1 dependent variable case.
    if Y.shape[1] == 1:
        return _pass_1d(
            X, Y[:, 0], W, obs_weights, tol,
            s_X, Xt_Y[:, 0], cho_Xt_X, max_iter, store_trace)

    # Handle multiple dependent variable case.
    else:

        # Holds regression weights.
        Ws, infos = [], []

        # Solve dependent variables sequentially.
        for i in range(Y.shape[1]):

            # Initial regression weights.
            _w = None if W is None else W[i]

            # Fit regression weights for dep var i.
            w_i, info = _pass_1d(
                X, Y[:, i], _w, obs_weights, tol,
                s_X, Xt_Y[:, i], cho_Xt_X, max_iter, store_trace)

            # Concatenate results.
            Ws.append(w_i)
            infos.append(info)

        return np.row_stack(Ws), infos


def _pass_1d(
        X, y, w, obs_weights, tol, s_X, Xt_y, cho_Xt_X, max_iter, store_trace):

    # Data dimensions.
    n_obs, n_feat = X.shape
    assert y.size == n_obs
    assert y.ndim == 1

    # Initialize weights.
    if w is None:
        w = cho_solve(cho_Xt_X, Xt_y - s_X)  # Newton step from origin.

    # Initial estimate for log-firing rates.
    Xw = X @ w
    Xw_norm = np.linalg.norm(Xw)

    # Store loss history.
    if store_trace:
        loss_hist = [np.sum(np.exp(Xw) - Xw * y)]
        approx_hist = [np.nan]
    else:
        loss_hist = []
        approx_hist = []

    # Now iteratively improve the regression weights
    # and the polynomial approximation.
    converged, itercount = False, 0
    while (not converged) and (itercount < max_iter):

        # Update polynomial approximation.
        (a2, a1, a0), resid = np.polyfit(
            Xw, np.exp(Xw), deg=2, full=True)[:2]

        # Update weights.
        w = (0.5 / a2) * cho_solve(cho_Xt_X, Xt_y - a1 * s_X)

        # Compute log firing rates.
        Xw_last = Xw
        Xw = X @ w

        # Check convergence of log-firing rates.
        if np.linalg.norm(Xw - Xw_last) < (tol * Xw_norm):
            converged = True
        else:
            Xw_norm = np.linalg.norm(Xw)

        # Store loss history if desired.
        if store_trace:
            loss_hist.append(
                obs_weights @ (np.exp(Xw) - Xw * y))
            approx_hist.append(
                obs_weights @ (a2 * Xw ** 2 + a1 * Xw + a0 - Xw * y))

        # Proceed to next iteration.
        itercount += 1

    # Return result with
    return w, Info(loss_hist, approx_hist, (a2, a1, a0))
