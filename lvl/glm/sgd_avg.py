"""
Stochastic Gradient Descent With Polyak Averaging
For Poisson Regression.
"""

import numpy as np
import numba
from lvl.glm.sgd import sgd_poiss_reg


def sgd_avg_poiss_reg(
        X, Y, W=None, obs_weights=None, ss=None,
        patience=4000, max_epochs=10, log_every=1000):

    # Use local Lipshitz constant for W == 0.
    if ss is None:
        ss = np.max(np.linalg.norm(X, axis=0)) ** -2

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

    # Run sgd without averaging until lost patience.
    W, init_loss_hist = sgd_poiss_reg(
        X, Y, W=W, obs_weights=obs_weights, init_ss=ss,
        num_ss=1, patience=patience, max_epochs=2,
        log_every=log_every)

    # Run sgd with averaging until lost patience.
    loss_sums = np.zeros(n_obs * max_epochs // log_every)
    loss_counts = np.zeros_like(loss_sums)

    W_avg = _sgd_polyak_avg(
        X, Y, W, obs_weights, ss, 3 * patience,
        max_epochs, log_every, loss_sums, loss_counts)

    # Determine average loss in each logging period.
    # If stopped early, filter out extra entries.
    idx = loss_counts > 0
    loss_hist = np.concatenate((
        init_loss_hist, loss_sums[idx] / loss_counts[idx]))

    return W_avg, loss_hist


@numba.jit(nopython=True)
def _sgd_polyak_avg(
        X, Y, W, obs_weights, ss, patience,
        max_epochs, log_every, loss_sums, loss_counts):

    # Initialize counters
    iter_count = 0       # Total iterations.
    patience_count = 0   # Number of iterations without progress.
    criterion = np.inf   # Progress threshold.
    ss_count = 0         # Number of step size decreases.
    epoch_count = 0      # Number of passes over the dataset.
    converged = False

    n_obs = X.shape[0]
    n_feat = X.shape[1]
    n_dep_vars = Y.shape[1]

    W_avg = np.zeros((n_dep_vars, n_feat))

    while (not converged) and (epoch_count < max_epochs):

        for i in np.random.permutation(n_obs):

            # Compute loss.
            Wx = W @ X[i]
            exp_Wx = np.exp(Wx)

            aWx = W_avg @ X[i]
            loss = obs_weights[i] * (np.sum(np.exp(aWx)) - (aWx @ Y[i]))

            alpha = iter_count / (iter_count + 1)
            beta = 1 - alpha

            # Implement stochastic gradient step.
            for j in range(n_dep_vars):
                for k in range(n_feat):
                    W[j, k] -= (
                        obs_weights[i] * ss *
                        (exp_Wx[j] - Y[i, j]) * X[i, k])

                    W_avg[j, k] = (
                        alpha * W_avg[j, k] + beta * W[j, k])

            # Progress observed, reset patience.
            if loss < criterion:
                patience_count = 0
                criterion = loss

            # Lost patience, decrease step size.
            elif patience_count > patience:
                converged = True
                break

            # Increment lost patience.
            else:
                patience_count += 1

            # Store the loss history.
            loss_sums[iter_count // log_every] += loss
            loss_counts[iter_count // log_every] += 1
            iter_count += 1

        # Count passes over the dataset.
        epoch_count += 1
