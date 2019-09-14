import numpy as np
import numba


class HMM:

    def __init__(self, n_components, observations="gaussian"):
        self.n_components = n_components
        self.observations = observations

    def fit(self, data, taus=None, maxiter=100):
        """
        Fits model parameters.

        Parameters
        ----------
        data : ndarray
            Has shape (n_timesteps, n_features).
        """

        params = _states(self.observations, self.n_components)

        # Optimization loop.
        for it in trange(maxiter):

            # E step.
            log_likes = params.likelihood(data)
            log_ps = forward_backward(log_transmat, log_likes)

            # M step.
            params = params.fit(log_ps, data)
            raise NotImplementedError()


@numba.jit(nopython=True)
def forward_backward(log_transmat, log_likes):
    """
    Implements Forward-Backward algorithm to identify the probability
    distribution of hidden states at each timestep, given a fixed
    transition matrix and liklihoods of observed data.

    Parameters
    ----------
    log_transmat : ndarray
        Log of transition matrix. Has shape (n_states, n_states).
    log_likes : ndarray
        Log likelihoods of each observation. Has shape
        (n_timesteps, n_states).

    Returns
    -------
    log_ps : ndarray
        Log probabilities of hidden state values. Has shape
        (n_timesteps, n_states).
    """

    # Allocate space for computation.
    log_fps = np.empty_like(log_likes)
    back_fps = np.empty_like(log_likes)

    # Forward pass.
    log_fps[0] = log_pi0 + log_likes[0]
    for t in range(1, n_time):

        # Compute log(transmat @ fps[t - 1]) and store result in log_fps[t].
        logmatvec(
            log_transmat, log_fps[t - 1], tmp_storage, log_fps[t])

        # Add contribution of log-likelihoods.
        log_fps[t] += log_likes[t]

    # Backward pass.
    log_bps[-1] = 0.0
    for t in range(1, n_time):

        # Compute log(transmat.T @ bps[-t]) and store result in log_bps[-t - 1]
        logmatvec(
            log_transmat.T, log_bps[-t], tmp_storage, log_bps[-t - 1])

        # Add contribution of log-likelihoods.
        log_bps[-t - 1] += log_likes[-t]

    # Normalize log-probabilities
    with objmode(log_ps='float64[:,:]'):
        log_ps = np.log(softmax(log_fps + log_bps, axis=1))

    return log_ps


@numba.jit(nopython=True)
def logmatvec(lgA, lgx, storage, out):
    """
    Computes matrix-vector multiplication in log space
    log(A @ x) given log(A) and log(x).

    Parameters
    ----------
    lgA : ndarray
        Log of m x n matrix
    lgx : ndarray
        Log of length-n vector.
    storage : ndarray
        ndarray same length as lgx. Overwritten with intermediate
        computations.
    out : ndarray
        ndarray same length as lgx. Overwritten with final
        result.
    """
    m, n = lgA.shape

    # Compute element i of the output vector.
    for i in range(m):

        # Compute terms in vector inner product (row i of A, and x).
        # Keeping track of the maximum value.
        mx = -np.inf
        for j in range(n):
            storage[j] = lgA[i, j] + lgx[j]
            if storage[j] > mx:
                mx = storage[j]

        # Compute numerically stable log-sum-exp of "storage".
        out[i] = 0.0
        for j in range(n):
            out[i] += np.exp(storage[j] - mx)
        out[i] = mx + np.log(out[i])
