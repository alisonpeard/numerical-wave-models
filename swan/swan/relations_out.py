import numpy as np
from numba import njit
from numpy.linalg import norm as l2
from .default_vars import *


def energy_density(N, Σ):
    """Calculate the energy density over spectral and geographic space.
    
    Parameters:
    -----------
    N : numpy.ndarray
        action density matrix of shape (nθ, nσ, ny, nx)
    Σ : numpy.ndarray
        action density np.meshgrid of shape (nθ, nσ, ny, nx)
    """
    Σ = Σ.reshape(N.shape, order=order)
    E = N * Σ
    return E


def action_density(E, Σ):
    """Calculate the energy density over spectral and geographic space.
    
    Parameters:
    -----------
    N : numpy.ndarray
        action density matrix of shape (nθ, nσ, ny, nx)
    Σ : numpy.ndarray
        action density np.meshgrid of shape (nθ, nσ, ny, nx)
    """
    Σ = Σ.reshape(N.shape, order=order)
    N = E / Σ
    return N


def sse_variance(E, dtheta, dsigma):
    """Calculate sea surface elevation variance over all components.
    
    AKA first moment of variance of SSE by integrating over sigma and theta.
    
    Units : m^2
    
    Parameters:
    -----------
    E : numpy.ndarray
        energy density matrix of shape (nθ, nσ, ...)
    dtheta : float
        constant direction delta
    dsigma : np.array
        log-distributed frequency delta
    """
    spectrum_1d = E.sum(axis=0) * dtheta
    m0 = (spectrum_1d * dsigma[:, np.newaxis]).sum(axis=0)
    # m0 = (spectrum_1d).sum(axis=0) * dsigma
    return m0
    
    
def total_energy(E, dtheta, dsigma, ρw=ρw, g=g):
    """Total energy across spatial grid."""
    m0 = sse_variance(E, dtheta, dsigma)
    Etot = 0.5 * ρw * g * m0
    return Etot


def significant_wave_height(E, dtheta, dsigma):
    """Significant wave height across spatial grid."""
    m0 = sse_variance(E, dtheta, dsigma)
    hs = 4 * np.sqrt(m0)
    return hs