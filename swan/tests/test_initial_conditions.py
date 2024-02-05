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


@pytest.mark.parametrize("f, n, kwargs", [(swan.cos2_model, 10, {'θm': 0}),
                                          (swan.cos2_model, 1000, {'θm': 0})])
def test_directional_speading(f, n, kwargs, limits=(-np.pi/2, np.pi/2)):
    """Test condition (6.3.20) from Holthuijsen (2007)
    
    (Integral over limits should be 1.)"""
    x = np.linspace(limits[0], limits[1], n)
    Δx = x[1] - x[0]
    integral = sum(f(x, **kwargs) * Δx)
    assert np.isclose(integral, 1.)