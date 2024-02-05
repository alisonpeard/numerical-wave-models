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
    

def test_dfdx_central_diff1():
    """Test for just (x, y)"""
    x = np.array([[1, 0, 1], [1, 0, 1], [2, 3, 2]])
    ny, nx = x.shape
    x = x.ravel(order=swan.order)
    stepy = 1
    stepx = stepy * ny
    n = len(x)
    Ay = swan.dfdx_central_diff(n, stepy, dx=0.5)
    xdy = (Ay @ x).reshape([ny, nx], order=swan.order)
    Ax = swan.dfdx_central_diff(n, stepx, dx=0.5)
    xdx = (Ax @ x).reshape([ny, nx], order=swan.order)
    x = x.reshape([ny, nx], order=swan.order)
    assert np.array_equal(xdy[1, :], x[-1, :] - x[0, :]), "ddx failed for columns"
    assert np.array_equal(xdx[:, 1], x[:, -1] - x[:, 0]), "ddx failed for rows"


def test_dfdx_central_diff2():
    """Test for matrix with θs."""
    thetas = np.array([-np.pi, -0.75 * -np.pi, -0.5 * -np.pi, -0.25 * -np.pi, 0,
                      0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi, np.pi])
    ntheta = len(thetas)
    x = np.array([[1, 0, 1], [1, 0, 1], [2, 3, 2]])
    ny, nx = x.shape
    stepy = ntheta
    stepx = stepy * ny
    x = x.ravel(order=swan.order)
    _, X = np.meshgrid(thetas, x, indexing='ij')
    X = X.ravel(order=swan.order)
    n = len(X)
    Ay = swan.dfdx_central_diff(n, stepy, dx=0.5)
    Xdy = (Ay @ X).reshape([ntheta, ny, nx], order=swan.order)
    Ax = swan.dfdx_central_diff(n, stepx, dx=0.5)
    Xdx = (Ax @ X).reshape([ntheta, ny, nx], order=swan.order)
    x = x.reshape([ny, nx], order=swan.order)
    for m in range(ntheta):
        assert np.array_equal(Xdy[m, 1, :], x[-1, :] - x[0, :]), "d/dx failed for columns"
        assert np.array_equal(Xdx[m, :, 1], x[:, -1] - x[:, 0]), "d/dx failed for rows"


def test_dfdm_central_diff():
    """Test for d/dm in m direction."""
    thetas = np.array([-np.pi, -0.75 * -np.pi, -0.5 * -np.pi, -0.25 * -np.pi, 0,
                      0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi, np.pi])
    dX = 3 * np.cos(thetas)
    ntheta = len(thetas)
    x = np.array([[1, 0, 1], [1, 0, 1], [2, 3, 2]])
    ny, nx = x.shape
    stepy = ntheta
    stepx = stepy * ny
    x = x.ravel(order=swan.order)
    θ, X = np.meshgrid(thetas, x, indexing='ij')
    X = X.ravel(order=swan.order)
    Am = swan.dfdm_central_diff(θ, stepx, stepy, dx=0.5, dy=0.5)
    Xdm = (Am @ X).reshape([ntheta, ny, nx], order=swan.order)
    assert np.allclose(Xdm[:, 1, 1], dX), "d/dm failed"