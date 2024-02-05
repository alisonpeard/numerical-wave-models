import numpy as np
from numba import njit
from numpy.linalg import norm as l2
from .default_vars import *
from .numerical import *


def relative_frequency(k, d, g=g):
    """Dispersion relation"""
    σ = np.sqrt(g * l2(k) * np.tanh(l2(k) * d))
    return σ


def absolute_frequency(σ, k, u):
    ω = σ + k @ u
    return ω


def inverse_dispersion_approx(ω, d, g=g):
    """Holthuijsen (2007) Equations (5.4.21) and (5.4.22).
    
    Exact in shallow and deep limits and erros less than 0.05% else."""
    k0 = ω**2 / g
    α = k0 * d
    β = α / np.sqrt((np.tanh(α)))
    k  = (α + β**2 * (np.cosh(β)**(-2)))
    k /= d * (np.tanh(β) + β * np.cosh(β)**(-2))
    return k


def phase_speed(k, ω):
    """
    Use dispersion relation to calculate the group velocity over a grid.
    
    This is the velocity that wave crests travels in the direction of the wave ray, s.
    
    c := ω / k
    """
    c = ω / k
    return c
                    
              
def group_velocity(k, d, ω):
    """
    Use dispersion relation to calculate the group velocity over a grid.
    
    This is the velocity that wave energy travels in the direction of the wave ray, s.
    
    cg = nc (Holthuijsen 2007)
    """
    n = (1 / 2) * (1 + (2 * k * d) / np.sinh(2 * k * d))
    c = phase_speed(k, ω)
    cg = n * c
    return cg


def spatial_group_velocity(cg, θ):
    """Transform propagation of wave energy into spatial (x,y)-space."""
    cx = cg * np.cos(θ)
    cy = cg * np.sin(θ)
    return np.array([cx, cy])


def frequency_velocity():
    """Shifting frequency due to variation in depth and currents."""
    # TODO
    pass


def directional_velocity(θ, c, cg, stepx, stepy, dx, dy):
    """Shifting direction (refraction) due to variation depth and currents.
    
    $$(-1 / k) * (dθ/dd * dd/dm + k * du/dm)$$
    """
    shape = c.shape
    c = c.ravel(order=order)
    cg = cg.ravel(order=order)
    A = dfdm_central_diff(θ, stepx, stepy, dx, dy)
    cθ = (-cg / c) * A @ c
    return cθ.reshape(shape, order=order)


