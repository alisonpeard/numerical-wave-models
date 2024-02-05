# pytest -x tests.py
from _pytest.fixtures import fixture
import pytest

try:
    import numpy as np
    import swan
except ImportError:
    pass


def test_imports():
    import numpy as np
    import swan


@pytest.mark.parametrize("dims", [[4, 4, 5, 2]])
def test_flattened_meshgrid(dims):
    ny, nx, nz0, nz1 = dims
    y = np.arange(0, ny)
    x = np.arange(0, nx)
    z = np.arange(0, nz0 * nz1).reshape([nz0, nz1])
    Y, X, Z = np.meshgrid(y, x, z.ravel(order=swan.order), indexing='ij')

    y2 = swan.flattened_meshgrid(Y, dims, which='y')
    x2 = swan.flattened_meshgrid(X, dims, which='x')
    z2 = swan.flattened_meshgrid(Z, dims, which='z')

    y2 = y2[::nz0,::nz1]
    x2 = x2[::nz1, ::nz0]
    z2 = z2[::ny, ::nx]

    assert np.array_equal(y2[:, 0], y)
    assert np.array_equal(x, x2[:, 0])
    assert np.array_equal(z, z2)