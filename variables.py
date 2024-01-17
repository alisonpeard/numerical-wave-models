import numpy as np
from numpy.linalg import norm as l2

# base variables to use
g = 9.81                  # acceleration due to gravity [ms-2]
d = 10                    # water depth [m]
u = np.array([0, 0])      # ambient current [ms-1]
λ = 80                    # wave length [m]
k = 2 * np.pi / λ         # wave number / spatial frequency of wave [m-1]
θ = np.pi / 4             # direction of wave (from East pointing)
k = np.array([k * np.cos(θ), k * np.sin(θ)])


def relative_frequency(k, d, g=g):
    """Dispersion relation"""
    σ = np.sqrt(g * l2(k) * np.tanh(l2(k) * d))
    return σ


def absolute_frequency(σ, k, u):
    ω = σ + k @ u
    return ω


def group_velocity(k, d, σ, u):
    """cg = (1 / 2) * (1 + (2 * l2(k) * d) / np.sinh(2 * l2(k) * d)) * (σ * k) / l2(k)**2 + u"""
    σ = σ[..., np.newaxis] # to allow broadcasting for ndarray σ
    cg = (1 / 2) * (1 + (2 * l2(k) * d) / np.sinh(2 * l2(k) * d))
    cg = cg[..., np.newaxis] * (σ * k) / l2(k)
    cg = cg + u
    return cg


def spatial_velocity(cg, u):
    velocity = cg + u
    cx = velocity[..., 0]
    cy = velocity[..., 1]
    return np.array([cx, cy])


def frequency_velocity():
    """Shifting frequency due to variation in depth and currents."""
    # TODO
    pass


def direction_velocity():
    """Shifting direction (refraction) due to variation depth and currents.
    
    $$(-1 / k) * (dθ/dd * dd/dm + k * du/dm)$$
    """
    # TODO
    pass


σ = relative_frequency(k, d)
cg = group_velocity(k, d, σ, u)
cx, cy = spatial_velocity(cg, u)