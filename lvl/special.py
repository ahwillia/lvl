import numpy as np
import numba


@numba.jit(nopython=True, cache=True)
def softplus(x):
    return np.log(1 + np.exp(x))
