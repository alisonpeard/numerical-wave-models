"""Base functions for numerical methods."""
import numpy as np
# from numba import njit
from scipy.linalg import block_diag
from .default_vars import *


def courant_number_2d(dx, dy, dt, u):
    C = (u[0] * dt / dx) + (u[1] * dt / dy)
    return C


# tested
def dfdx_central_diff(n, stepx, dx=0.5):
    A = np.diag((1 / (2 * dx)) * np.ones(n - stepx), stepx) - np.diag((1 / (2 * dx)) * np.ones(n - stepx), -stepx)
    return A


def frequency_grid(nsigma, σmin=σmin, σmax=σmax, scale='linear'):
    """Make frequenct grid using specified scale."""
    if scale == 'linear':
        sigmas = np.linspace(σmin, σmax, nsigma)
        dsigma = (sigmas[1] - sigmas[0]) * np.ones(nsigma)
    elif scale == 'log':
        sigmas = np.geomspace(σmin, σmax, nsigma)
        d = sigmas[1:] - sigmas[:-1]
        d_const = (d / sigmas[1:])[0]
        dsigma = sigmas.copy()
        dsigma[1:] = d
        dsigma[0] *= d_const
    else:
        raise ValueError("'scale' must be 'linear' or 'log'")
    return sigmas, dsigma

# spectral space
# tested
def dfdm_central_diff(direction, stepx, stepy, dx=0.5, dy=0.5):
    """Derivation from SWAN Eq (3.82)
    https://swanmodel.sourceforge.io/online_doc/swantech/node54.html
    """
    θ = direction.ravel(order=order)
    n = len(θ)
    left = dfdx_central_diff(n, stepx, dx) * np.sin(θ)[:, np.newaxis]
    right = dfdx_central_diff(n, stepy, dy) * np.cos(θ)[:, np.newaxis]
    A = right - left
    return A


def dfds_central_diff(direction, stepx, stepy, dx=0.5, dy=0.5):
    """Double check derivation from SWAN Eq (3.82)
    
    https://swanmodel.sourceforge.io/online_doc/swantech/node54.html
    """
    θ = direction.ravel(order=order)
    n = len(θ)
    left = dfdx_central_diff(n, stepx, dx) * np.cos(θ)[:, np.newaxis]
    right = dfdx_central_diff(n, stepy, dy) * np.sin(θ)[:, np.newaxis]
    A = left + right
    return A


# numerical fixes
def eliminate_negative_energies(E, dtheta, ntheta):
    """Eliminate negative energy densities.
    
    Conservative and strict elimination (Tolman, 1991), SWAN (3.27)."""
    Epos = np.where(E > 0, E, 0)

    Etot = E.sum(axis=0) * dtheta
    Eptot = Epos.sum(axis=0) * dtheta

    conservative = Eptot != 0  # leave as 1 if 0/0
    strict = Etot >= 0         # strict elimination: alpha=1 when Etot is negatives
    alpha = np.divide(Etot, Eptot, out=np.ones_like(Etot), where=(conservative & strict))
    alpha = np.repeat(alpha[np.newaxis, :], repeats=ntheta, axis=0)

    return Epos * alpha


# directional sweeps
def flip_for_sweep(N, sweep):
    if sweep == 0: # first quadrant
        return N
    elif sweep == 1: # second quadrant (flip x)
        return N[..., :, ::-1]
    elif sweep == 2: # third quadrant (flip x and y)
        return N[..., ::-1, ::-1]
    elif sweep == 3: # fourth quadrant (flip y)
        return N[..., ::-1, :]
    else:
        raise ValueError("Sweep number must be one of [0, 1, 2, 3]")


# def filter_cxcy_for_sweep(cx, cy, sweep):
#     if sweep == 1:
#         condition = (cx > 0) & (cy > 0)
#     elif sweep == 2:
#         condition = (cx < 0) & (cy > 0)
#     elif sweep == 3:
#         condition = (cx < 0) & (cy < 0)
#     elif sweep == 4:
#         condition = (cx > 0) & (cy < 0)
#     else:
#         raise ValueError("Sweep number must be one of [1, 2, 3, 4]")
#     cx = np.where(condition, cx, 0)
#     cy = np.where(condition, cy, 0)
#     return cx, cy