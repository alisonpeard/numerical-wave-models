"""Base functions for numerical methods."""
import numpy as np
# from numba import njit
from scipy.linalg import block_diag
from .default_vars import *


def courant_number_2d(dx, dy, dt, u):
    C = (u[0] * dt / dx) + (u[1] * dt / dy)
    return C


# # tested
# def dfdx_central_diff(n, stepx=1, dx=0.5):
#     A = np.diag((1 / (2 * dx)) * np.ones(n - stepx), stepx) - np.diag((1 / (2 * dx)) * np.ones(n - stepx), -stepx)
#     return A


def hybrid(c, mu=.5):
    """Create 1D first order hybrid scheme finite difference matrix."""
    assert (mu >= 0) & (mu <=1), "mu must be in range [0, 1]."
    if type(c) == int:
        c = np.ones(c)
    elif len(c) > 0:
        c = c.ravel(order=order)
    c_pos = np.where(c >= 0, c, 0)
    c_neg = np.where(c < 0, c, 0)
    A = mu * central_diff(c) + (1 - mu) * (upwind(c_pos) + upwind_r(c_neg))
    return A


def central_diff(n):
    """Create 1D first order central difference finite difference matrix."""
    if type(n) == int:
        n = np.ones(n)
    elif len(n) > 0:
        n = n.ravel(order=order)
    A = 0.5 * np.diag(n[:-1], 1) - 0.5 * np.diag(n[1:], -1)
    return A


def upwind(n):
    """Create 1D first order upwind scheme finite difference matrix.
    
    For positive velocities c >= 0."""
    if type(n) == int:
        n = np.ones(n)
    elif len(n) > 0: # ie array-like
        n = n.ravel(order=order)
    A = np.diag(n) - np.diag(n[1:], -1)
    return A


def upwind_r(n):
    """Create 1D first order upwind scheme finite difference matrix.
    
    For negative velocities c < 0."""
    if type(n) == int:
        n = np.ones(n)
    elif len(n) > 0: # ie array-like
        n = n.ravel(order=order)
    A = np.diag(n[:-1], 1) - np.diag(n)
    return A


# def hybrid(c, mu=.5):
#     """
#     Create 1D first order hybrid scheme finite difference matrix.
    
#     Parameters:
#     -----------
#     c : numpy.ndarray
#         Vector of velocity coefficients.
#     mu : float (0, 1)
#         Blending parameter. mu=0: upwind scheme, mu=1: central
#         difference scheme.
#     """
#     c = c.ravel(order=order).copy()
#     positive_dir = c >= 0
    
#     coeff_diag = np.where(positive_dir, 1 - 0.5 * mu, 0.5 * mu)
#     coeff_super = np.where(positive_dir, 0.5 * mu, 1 - 0.5 * mu)
#     left = coeff_diag * np.diag(c)
#     left += coeff_super * np.diag(c[:-1], 1)
    
#     coeff_diag = np.where(positive_dir, 0.5 * mu, 1 - 0.5 * mu)
#     coeff_sub = np.where(positive_dir, 1 - 0.5 * mu, 0.5 * mu)
#     right = coeff_diag * np.diag(c[1:], -1)
#     right += coeff_sub * np.diag(c)
    
#     A = left - right
#     return A


def kronecker_product(Ax, Ay, cx=1, cy=1):
    """Use Kronecker product to construct multidimensional finite-difference matrix.

    Assuming column-major, so y-axis varies fastest.

    https://math.mit.edu/~stevenj/18.303/lecture-10.html
    """
    Ix = np.eye(*Ax.shape)
    Iy = np.eye(*Ay.shape)
    A = (np.kron(Ix, Ay) * cy) + (np.kron(Ax, Iy) * cx) # same as np.diag(cx) @ (...) but allows for scalars
    return A


def frequency_grid(nsigma, σmin=σmin, σmax=σmax, scale='linear'):
    """Make frequency grid using specified scale."""
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
    
    Conservative and strict elimination (Tolman, 1991), SWAN (3.27)
    
    Parameters:
    -----------
    E : np.ndarray
        Numpy array of energy densities of dimensio (nθ,nσ,...).
    dtheta : float
    nthete : int
    """
    Epos = np.where(E > 0, E, 0)

    Etot = E.sum(axis=0) * dtheta
    Eptot = Epos.sum(axis=0) * dtheta

    conservative = Eptot != 0  # leave as 1 if 0/0
    strict = Etot >= 0         # strict elimination: alpha=1 when Etot is negatives
    alpha = np.divide(Etot, Eptot, out=np.ones_like(Etot), where=(conservative & strict))
    alpha = np.repeat(alpha[np.newaxis, :], repeats=ntheta, axis=0)

    return Epos * alpha


# utils for directional sweeps
def sweep_flip(N, sweep):
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


def flip_spatial_velocities(cx, cy, sweep):
    """Double-check this makes sense."""
    if sweep == 0:
        return cx, cy
    elif sweep == 1:
        return -cx, cy
    elif sweep == 2:
        return -cx, -cy
    elif sweep == 3:
        return cx, -cy
    else:
        raise ValueError("Sweep number must be one of [0, 1, 2, 3]")
    

def flip_directional_velocities(cθ, sweep):
    """Double-check this makes sense."""
    if sweep == 0:
        return cθ
    elif sweep == 1:
        return -cθ
    elif sweep == 2:
        return cθ
    elif sweep == 3:
        return -cθ
    else:
        raise ValueError("Sweep number must be one of [0, 1, 2, 3]")
    

# helper functions
def indicate_boundaries2d(ny, nx):
    """Return mask indicating boundary cells for x and y."""
    xmin = np.zeros([ny, nx], dtype=bool).ravel(order=order)
    xmin[:ny] = 1
    
    xmax = np.zeros([ny, nx], dtype=bool).ravel(order=order)
    xmax[-ny:] = 1
    
    ymin = np.zeros([ny, nx], dtype=bool).ravel(order=order)
    ymin[::nx] = 1
    
    ymax = np.zeros([ny, nx], dtype=bool).ravel(order=order)
    ymax[(nx-1)::nx] = 1
    return xmin, xmax, ymin, ymax