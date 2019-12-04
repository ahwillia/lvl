import numpy as np
from scipy.optimize import minimize


def sgd_poiss_reg(
        X, Y, W=None, obs_weights=None, init_ss=None, num_ss=5,
        patience=500, max_epochs=10, log_every=1000):

    # Use local Lipshitz constant for W == 0.
    if init_ss is None:
        init_ss = np.max(np.linalg.norm(X, axis=0)) ** -2

    # Determine data dimensions.
    Y = Y[:, None] if (Y.ndim == 1) else Y
    n_obs, n_features = X.shape
    n_dep_vars = Y.shape[1]
    assert Y.shape[0] == n_obs

    # Initialize weights.
    if W is None:
        W = np.zeros((n_dep_vars, n_features))

    # Set observation weights.
    if obs_weights is None:
        obs_weights = np.ones(n_obs)

    # Allocate space for loss history.
    loss_sums = np.zeros(n_obs * max_epochs // log_every)
    loss_counts = np.zeros_like(loss_sums)

    # Run SGD.
    _sgd_solver(
        X, Y, W, obs_weights, init_ss, num_ss, patience,
        max_epochs, log_every, loss_sums, loss_counts)

    # Determine average loss in each logging period.
    # If stopped early, filter out extra entries.
    idx = loss_counts > 0
    loss_hist = loss_sums[idx] / loss_counts[idx]

    return W, loss_hist


def scipy_poiss_reg(X, y, method, W=None, obs_weights=None, verbose=False):

    # Fit multiple dependent variables sequentially.
    y = np.squeeze(y)

    if y.ndim > 1:

        W_fitted = []
        for p in range(y.shape[1]):

            init_W = np.zeros(X.shape[1]) if W is None else W[p]

            W_fitted.append(scipy_poiss_reg(
                X, y[:, p], method,
                W=init_W,
                obs_weights=obs_weights,
                verbose=verbose)[0])

        return np.row_stack(W_fitted), np.array([])

    # initialize weights for each observation.
    ow = np.ones_like(y) if obs_weights is None else obs_weights

    # loss function and gradient
    def f(w):
        Xw = X @ w
        exp_Xw = np.exp(Xw)
        loss = ow @ (exp_Xw - y * Xw)
        grad = X.T @ (ow * (exp_Xw - y))
        return loss, grad

    # hessian
    def hess(w):
        Xw = (ow * np.exp(X @ w))
        return X.T @ (Xw[:, None] * X)

    # optimize
    init_W = np.zeros(X.shape[1]) if W is None else W
    result = minimize(
        f, init_W, jac=True, hess=hess, method=method)

    if verbose:
        print(result.message)

    return result.x, np.array([])
