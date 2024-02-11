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