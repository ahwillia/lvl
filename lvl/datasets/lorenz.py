"""
Toy synthetic dataset based on the Lorenz attractor.
"""
import numpy as np

from lvl.utils import rand_orth, get_random_state


def poisson_lorenz(
        n_out, n_steps, x0=None, dt=0.01, latent_noise_scale=10.0,
        max_rate=10.0, min_rate=0.01, seed=None):
    """
    Simulate high-dimensional count data series following
    low-dimensional Lorenz attractor dynamics.

    Parameters
    ----------
    n_out : int
        Dimensional of observations.
    n_steps: int
        Number of observed timesteps.
    dt : float
        Euler integration step of the continuous time
        ODE.
    latent_noise_scale : float
        Scale of Wiener process noise on latent states.
        Note that the square root of dt also scales
        this noise source (Euler–Maruyama integration).
    max_rate : float
        Maximum rate parameter in the simulated
        dataset.
    min_rate : float
        Minimum rate parameter in the simulated
        dataset.
    seed : None, int, or np.random.RandomState
        Seed for random number generator.

    Returns
    -------
    data : ndarray
        Data array holding simulated count data. Has shape
        (n_steps, n_out).
    rates : ndarray
        True time-varying rate parameters, associated with
        'data'. Has shape (n_steps, n_out).
    W : ndarray
        Weight matrix. Has shape (n_out, 3).
    X : ndarray
        Simulated latent states. Has shape (n_steps, 3).
    """

    # Initialize random number generator.
    rs = get_random_state(seed)

    # Parameters of Lorenz equations (chaotic regime).
    sigma = 10.0
    beta = 8 / 3
    rho = 28.0

    # Allocate space for simulation.
    x = x0 if x0 is not None else np.ones(3)
    dxdt = np.empty(3)
    x_hist = np.empty((n_steps, 3))

    # Draw random readout matrix.
    W = rand_orth(3, n_out, seed=rs)

    # Simulate latent states.
    for t in range(n_steps):

        # Lorenz equations
        dxdt[0] = sigma * (x[1] - x[0])
        dxdt[1] = x[0] * (rho - x[2]) - x[1]
        dxdt[2] = x[0] * x[1] - beta * x[2]

        # Euler–Maruyama integration
        eta = latent_noise_scale * rs.randn(3)
        x = x + (dt * dxdt) + (np.sqrt(dt) * eta)

        # Store latent variable traces
        x_hist[t] = x

    # Center the x's so they exert comparable effects
    # in the observed data.
    x_hist = x_hist - np.mean(x_hist, axis=0)

    # Rescale rates to desired range.
    log_rates = np.dot(x_hist, W)
    log_rates = \
        (log_rates - np.min(log_rates)) / np.ptp(log_rates)
    log_rates = \
        log_rates * np.log(max_rate / min_rate) + np.log(min_rate)
    rates = np.exp(log_rates)

    # Draw Poisson distributed observations.
    data = rs.poisson(rates)

    # Return quantities of interest.
    return data, rates, W, x_hist
