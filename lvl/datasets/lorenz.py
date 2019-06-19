"""
Toy synthetic dataset based on the Lorenz attractor.
"""
import numpy as np

from ..utils import rand_orth, get_random_state


def simulate_lorenz(
        n_out, n_steps, x0=None, dt=0.01, latent_noise_scale=10.0,
        readout_weights_scale=1e-1, readout_weights="orthogonal",
        observations="poisson", seed=None):
    """
    Simulate high-dimensional data following low-dimensional
    Lorenz attractor dynamics.

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
    readout_weights_scale : float
        Scale of readout weight parameters. The Frobenius
        norm of the weight matrix is scaled to equal
        this parameter.
    readout_weights : str
        Specifies form of the readout weights. Options
        are ("orthogonal", "randn", "rand") corresponding
        to a random orthogonal matrix, random gaussian
        weights, random uniform weights on the interval
        [0, 1).
    observations : str
        Specifies the form of the observations. Options
        are ("poisson",).
    seed : None, int, or np.random.RandomState
        Seed for random number generator.

    Returns
    -------
    data : ndarray
        Data array holding simulated data. Has shape
        (n_steps, n_out).
    denoised : ndarray
        Data array in absence of noise. For Poisson
        distributed observations, these are the
        log firing rates. Has shape (n_steps, n_out).
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
    if readout_weights == "orthogonal":
        W = rand_orth(3, n_out, seed=rs)

    elif readout_weights == "randn":
        W = rs.randn(3, n_out)

    elif readout_weights == "rand":
        W = rs.rand(3, n_out)
    else:
        raise ValueError(
            "Did not recognize 'readout_weights' option.")

    # Rescale output matrix to desired norm.
    W *= readout_weights_scale / np.linalg.norm(W)

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

    # Compute data in absence of noise.
    denoised = np.dot(x_hist, W)

    # Add noise.
    if observations == "poisson":
        data = rs.poisson(np.exp(denoised))
    else:
        raise ValueError(
            "Did not recognize 'observations' option.")

    # Return quantities of interest.
    return data, denoised, W, x_hist
