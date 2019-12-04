"""
PASS-GLM

Code from David Zoltowski, with minor modifications.
 - Repo: https://github.com/davidzoltowski/paglm/
 - License: MIT

Added features include:
 - Fully automated interval selection
 - Weighted observations in the log-likelihood

References
----------
Zoltowski & Pillow (2018). Scaling the Poisson GLM to massive neural datasets
through polynomial approximations. NeurIPS.

Huggins et al. (2017). PASS-GLM: polynomial approximate sufficient statistics
for scalable Bayesian GLM inference. NeurIPS.
"""
import numpy as np
from scipy.special import gammaln
from lvl.utils import lsqn
from lvl.special import softplus

SuffStats = namedtuple('SuffStats', ['x', 'yx', 'xxt', 'yxxt'])
Interval = namedtuple('Interval', ['lower', 'upper', 'X_sub', 'Y_sub'])


def pass_poiss_reg(
        X, Y, method, W=None, obs_weights=None, f="exp",
        subset_frac=0.1, suff_stats=None, interval=None):

    # Compute suff stats.
    if suff_stats is None:
        suff = []
        suff += [np.sum(X, axis=0)]
        suff += [X.T @ y]
        suff += [X.T @ X]
        suff += [X.T @ (X.T @ y)] if f == "softplus" else [None]
        suff = SuffStats(*suff)

    # Determine interval for approximation.
    if interval is None:
        Interval()

        suff_stats = process_data(X, y, f=f, subset_frac=subset_frac)

    # Fit
    w_paglm, interval = fit_paglm(
        f, suff, dtSp, intervals, Cinv=Cinv, X_sub=X_sub, y_sub=y_sub)

    W = fit_paglm(np.exp, suff, dt, intervals, Cinv=None, X_sub=0, y_sub=0)

    return W, np.array([])


def poisson_log_like(
        w, Y, X, dt, f=np.exp, Cinv=None):
    """
    Poisson GLM log likelihood.
    f is exponential by default.
    """

    # if no prior given, set it to zeros
    if Cinv is None:
        Cinv = np.zeros([np.shape(w)[0], np.shape(w)[0]])

    # evaluate log likelihood and gradient
    ll = 0.5 * (w.T @ Cinv @ w) + np.sum(
        Y * np.log(f(X @ w)) - f(X @ w)*dt - gammaln(Y + 1) + Y*np.log(dt))

    return ll


def process_data(
        X, y, X_sub=None, y_sub=None, insuff=None, subset_frac=0.1):
    """
    This function builds and returns the quadratic sufficient statistics, and
    stores a specified subset of the data.

    The sufficient statistics are:
        suff[0] = sum x
        suff[1] = sum y*x
        suff[2] = sum x*x^T
        suff[3] = sum y*x*x^T

    The subset of data is stored in X_sub and y_sub, and the fraction to be
    stored is given by subset_frac.
    """

    if insuff is not None:
        suff = [x + y for x, y in zip(suff, insuff)]

    # num data
    T = np.shape(y)[0]

    # get random subset of data
    indices = np.random.choice(T, int(T*subset_frac), replace=False)
    if X_sub is None:
        X_sub = X[indices, :]
        y_sub = y[indices]
    else:
        X_sub = np.vstack([X_sub, X[indices, :]])
        y_sub = np.concatentate([y_sub, y[indices]])

    return suff, X_sub, y_sub


def adaptive_interval(f, suff, Xs, ys, dt, intervals, Cinv=None):
    """
    This function performs adaptive interval selection given a subset of data
    Xs and ys. It takes as input the nonlinearity f, the quadratic sufficient
    statistics suff, Xs, ys, the bin size dt, a list of intervals, and a prior.

    It outputs the weights computed using selected approximation interval and
    the identified approximation interval.
    """

    # allocate zeros for log like, one for each interval
    log_likes = np.zeros([len(intervals)])

    # for each input interval
    for idx, interval in enumerate(intervals):

        # compute paglm weights
        w_paglm = compute_paglm_weights(f, suff, interval, dt, Cinv=Cinv)

        # evaluate log likelihood of subset of data
        log_likes[idx] = poisson_log_like(w_paglm, ys, Xs, dt, f=f, Cinv=Cinv)

        # store weights if they are the highest log-like so far
        if np.argmax(log_likes[:idx+1]) == idx:
            w_star = w_paglm

    # return weights and best interval
    max_interval = intervals[np.argmax(log_likes)]

    return w_star, max_interval


def fit_paglm(f, suff, dt, intervals, Cinv=None, X_sub=0, y_sub=0):
    """
    This function fits the paGLM model to data X, y with function f and
    the designated interval.

    If intervals has length one, then the function uses a fixed interval
    for the approximation. Otherwise, the function chooses the interval
    that maximizes the log-likelihood of a subset of the data (adaptive).
    """

    # fixed interval
    if intervals is None or len(intervals) 
    if len(intervals) == 1:

        w_star = compute_paglm_weights(f, suff, intervals[0], dt, Cinv=Cinv)

    # adaptive
    else:

        # adapt interval
        w_star, interval = adaptive_interval(
            f, suff, X_sub, y_sub, dt, intervals, Cinv=Cinv)

    # return weights and interval used for approximation
    return w_star, interval


def compute_paglm_weights(f, suff, interval, dt, Cinv=None):
    """
    This function computes the paglm weights given a nonlinearity, sufficient
    statistics, bin size, and optional prior over an approximation interval.
    """

    # if Cinv is none, make prior a matrix of zeros
    if Cinv is None:
        Cinv = np.zeros((np.shape(w)[0], np.shape(w)[0]))

    # compute Chebyshev approximation and paGLM weights
    a0, a1, a2 = compute_chebyshev(f, interval, power=2, dx=0.01)

    # if exponential, compute weights with only one approximation
    if f == "exp":
        w_paglm = lsqn(
            2.0 * a2 * dt * suff[2] + Cinv,
            suff[1] - suff[0] * a1 * dt)

    # if not exp, compute second approximation and the resulting weights
    else:
        assert f == "softplus"
        b0, b1, b2 = chebyshev_coeffs(f, interval, power=2, dx=0.01)
        w_paglm = lsqn(
            2.0 * (a2 * dt * suff[2] - b2 * suff[3]) + Cinv,
            suff[1] * b1 - suff[0] * a1 * dt)

    # return weights
    return w_paglm


def chebyshev_coeffs(f, xlim, power=2, dx=0.01):
    """
    Compute Chebyshev polynomial approximations to functions.
    Uses weighted linear regression.

    Inputs:
    f: str, either "exp" or "softplus"
    xlim: interval of approximation
    power: order of approximation
    dx: discretization for approximation

    Output:
    what_cheby: weights of approximation, in increasing order
    """
    if f == "exp":
        func = np.exp
    elif f == "softplus":
        func = softplus

    # grid approximation range
    xx = np.arange(xlim[0] + dx/2.0, xlim[1], dx)
    nx = xx.shape[0]
    xxw = np.arange(
        -1.0 + (1 / nx), 1.0, 2.0 / nx)  # relative locations in [-1,1]

    # create monomial basis
    Bx = np.zeros([nx, power + 1])
    for i in range(0, power + 1):
        Bx[:, i] = np.power(xx, i)

    # compute weighting for weighted linear regression
    errwts_cheby = 1.0 / np.sqrt(1 - xxw ** 2)
    Dx = np.diag(errwts_cheby)

    # evaluate function on grid
    fx = func(xx)

    # compute approximation weights (in monomial basis)
    what_cheby = lstsqn(Bx.T @ Dx @ Bx, Bx.T @ Dx @ fx)

    return what_cheby
