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
    

@pytest.fixture
def simple_linear_system():
    """Burden and Faires Section 7.3 Example 1."""
    A = np.array([[10, -1, 2, 0],
                  [-1, 11, -1, 3],
                  [2, -1, 10, -1],
                  [0, 3, -1, 8]], dtype=float)
    b = np.array([6, 25, -11, 15], dtype=float)
    x0 = np.zeros(4)
    x = np.array([1, 2, -1, 1], dtype=float)
    return A, b, x0, x


@pytest.mark.parametrize("linear_system", ['simple_linear_system'])
def test_gauss_seidel(linear_system, request):
    A, b, x0, x = request.getfixturevalue(linear_system)
    tol = 1e-8
    x_gs = swan.gauss_seidel(A, b, x0, tol=tol)
    assert np.allclose(x_gs, x, atol=tol)


@pytest.mark.parametrize("linear_system", ['simple_linear_system'])
def test_exact(linear_system, request):
    A, b, x0, x = request.getfixturevalue(linear_system)
    tol = 1e-8
    x_exact = swan.exact_solution(A, b)
    assert np.allclose(x_exact, x, atol=tol)