# pytest -x tests.py
# pytest -x tests/*
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


@pytest.mark.parametrize("nx, ny", [(2, 2)])
def test_bsbt_matrix(nx, ny, cy=1, cx=1, dy=1, dx=1):
    """Test BSBT matrix being constructed correctly."""
    n = nx * ny
    cy = cy * np.ones(n)
    cx = cx * np.ones(n)
    stepy = 1
    stepx = nx
    dy = 1
    dx = 1
    A = swan.bsbt_matrix(stepx, stepy, cx, cy, dx, dy)
    assert np.diag(A).sum() == (cx + cy).sum()
    assert np.diag(A, -stepy).sum() == -cy[stepy:].sum()
    assert np.diag(A, -stepx).sum() == -cx[stepx:].sum()


def test_bsbt_energy_conservation1d():
    """Test that all intial flux leaves through the out-boundary for x-propagation."""
    nx = 10
    dt = 1
    dx = 1
    cx = 0.2
    T = 200
    A = swan.bsbt_matrix(1, 0, cx * np.ones(nx), np.array([0]), dx, dx)
    N = np.zeros([nx, 1])
    N[0] = 0.1
    N_mat = np.zeros([nx, T + 1])
    N_mat[:, 0] = N[:,0]
    for t in range(T):
        N = N - dt * (A @ N)
        N_mat[:, t + 1] = N.squeeze()
    assert np.isclose(N_mat[0, :].sum(), N_mat[-1, :].sum()), "Flux not conserved in x-propagation."