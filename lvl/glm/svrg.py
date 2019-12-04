"""
Stochastic Variance-Reduced Gradient (SVRG) Method for Poisson
Regression.

Reference
---------
Harikandeh et al (2015). Stop wasting my gradients: Practical SVRG.
"""

import numpy as np
import numba


def svrg_poiss_reg(
        X, Y, W=None, batch_size=100, n_updates=1000, obs_weights=None,
        init_ss=None, num_ss=5, patience=500, max_epochs=10, log_every=1000):

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

    # Run SVRG solver (overwrites W with result).
    _svrg(
        X, Y, W, obs_weights, batch_size, n_updates, init_ss, num_ss,
        patience, max_epochs, log_every, loss_sums, loss_counts)

    # Determine average loss in each logging period.
    # If stopped early, filter out extra entries.
    idx = loss_counts > 0
    loss_hist = loss_sums[idx] / loss_counts[idx]

    return W, loss_hist


@numba.jit(nopython=True)
def _svrg(
        X, Y, W, obs_weights, batch_size, n_updates, ss, num_ss, patience,
        max_epochs, log_every, loss_sums, loss_counts):

    # Initialize counters
    iter_count = 0       # Total iterations.
    patience_count = 0   # Number of iterations without progress.
    criterion = np.inf   # Progress threshold.
    ss_count = 0         # Number of step size decreases.
    epoch_count = 0      # Number of passes over the dataset.
    converged = False

    n_obs = X.shape[0]
    n_features = X.shape[1]
    n_dep_vars = Y.shape[1]

    snapW = np.zeros((n_dep_vars, n_features))
    snap_grad = np.zeros((n_dep_vars, n_features))

    prm = np.arange(n_obs)

    # Main Loop.
    while (not converged) and (epoch_count < max_epochs):

        # Shuffle order.
        np.random.shuffle(prm)

        # Save weights.
        snapW[:] = W
        snap_grad.fill(0.0)

        # Compute snapshot gradient.
        for _i in range(batch_size):

            # Sample datapoint without replacement
            i = prm[_i]

            # Compute loss.
            Wx = W @ X[i]
            exp_Wx = np.exp(Wx)

            # Add contribution to snapshot gradient.
            z = obs_weights[i] / batch_size

            for j in range(n_dep_vars):
                for k in range(n_features):
                    snap_grad[j, k] += \
                        z * (exp_Wx[j] - Y[i, j]) * X[i, k]

        # Stochastic update iterations.
        for _ in range(n_updates):

            # Sample observation. Technically, should be done
            # without replacement.
            _i = np.random.randint(n_obs)
            i = prm[_i]

            # Compute loss.
            Wx = W @ X[i]
            exp_Wx = np.exp(Wx)
            loss = obs_weights[i] * (np.sum(exp_Wx) - (Wx @ Y[i]))

            # If observation was used for snapshot gradient, just
            # do an sgd step.
            if _i < batch_size:

                # Implement stochastic gradient step.
                for j in range(n_dep_vars):
                    for k in range(n_features):
                        W[j, k] -= (
                            obs_weights[i] * ss *
                            (exp_Wx[j] - Y[i, j]) * X[i, k])

            # Use combination of local gradient and snapshot.
            else:

                # Implement variance-reduced gradient step.
                exp_snapWx = np.exp(snapW @ X[i])

                for j in range(n_dep_vars):
                    for k in range(n_features):

                        # Compute difference with snapshot gradient.
                        g1 = (exp_Wx[j] - Y[i, j]) * X[i, k]
                        g2 = (exp_snapWx[j] - Y[i, j]) * X[i, k]
                        g_diff = obs_weights[i] * (g1 - g2)

                        # Update weights.
                        W[j, k] -= ss * (g_diff + snap_grad[j, k])
                        # W[j, k] -= ss * g1
                        # W[j, k] -= ss * g2
                        # W[j, k] -= ss * snap_grad[j, k]

            # Progress observed, reset patience.
            if loss < criterion:
                patience_count = 0
                criterion = loss

            # Lost patience, decrease step size.
            elif patience_count > patience:

                ss_count += 1

                if ss_count >= num_ss:
                    converged = True
                    break
                else:
                    patience_count = 0
                    patience *= 2
                    ss *= 0.5

            # Increment lost patience.
            else:
                patience_count += 1

            # Store the loss history.
            loss_sums[iter_count // log_every] += loss
            loss_counts[iter_count // log_every] += 1
            iter_count += 1

        # Count passes over the dataset.
        epoch_count += 1
