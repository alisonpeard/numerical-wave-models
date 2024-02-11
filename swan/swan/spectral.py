import numpy as np
# from numba import njit
from scipy.linalg import block_diag
from .default_vars import *
from .numerical import *


def hybrid_matrix_theta(cθ, step=1, nu=0.5):
    """nu=0: upwind scheme, nu=1: central difference scheme.
    step should be 1 for row-major ('C') ordering in (σ,θ).
    """
    cθ = cθ.ravel(order=order).copy() # numpy arrays are mutable
    counter_clockwise = cθ >= 0
    
    coeff_diag = np.where(counter_clockwise, 1 - 0.5 * nu, 0.5 * nu)
    coeff_super = np.where(counter_clockwise, 0.5 * nu, 1 - 0.5 * nu)
    left = coeff_diag * np.diag(cθ)
    left += coeff_super * np.diag(cθ[:-step], step)
    
    coeff_diag = np.where(counter_clockwise, 0.5 * nu, 1 - 0.5 * nu)
    coeff_sub = np.where(counter_clockwise, 1 - 0.5 * nu, 0.5 * nu)
    right = coeff_diag * np.diag(cθ[step:], -step)
    right += coeff_sub * np.diag(cθ)
    
    A = left - right
    return A


def theta_block_matrix(c, step, nu=0.5, axis=-1):
    n = c.shape[axis]
    grid_cells = np.split(c, n, axis=axis)
    blocks = [hybrid_matrix_theta(grid_cell, step, nu) for grid_cell in grid_cells]
    A = block_diag(*blocks)
    return A
