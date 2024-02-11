import numpy as np
# from numba import njit
from scipy.linalg import block_diag
from .default_vars import *
from .numerical import *

# geographic space
def upwind(n):
    """Create 1D first order upwind scheme finite difference matrix."""
    if type(n) == int:
        A = np.eye(n) - np.diag(np.ones(n-1), -1)
    elif len(n) > 0: # ie array-like
        n = n.ravel(order=order)
        A = np.diag(n) - np.diag(n[1:], -1)
    return A


