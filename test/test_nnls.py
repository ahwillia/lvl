import pytest
from numpy.testing import assert_allclose
from lvl.nnls import nnls
import numpy as np

SEED = 1234


@pytest.mark.parametrize("method", ["cd"])
@pytest.mark.parametrize("m", [10, 20])
@pytest.mark.parametrize("n", [1, 5])
@pytest.mark.parametrize("k", [1, 5])
def test_self_consistency(method, m, n, k):

    rs = np.random.RandomState(SEED)

    A = rs.randn(m, n)
    B = rs.randn(m, k)

    # Use scipy as a gold standard.
    X_sci = nnls(A, B, method="scipy")

    X = nnls(A, B, method=method)
    assert_allclose(X, X_sci)
