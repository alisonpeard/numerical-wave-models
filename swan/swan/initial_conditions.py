import numpy as np
from numba import njit
from numpy.linalg import norm as l2
from .default_vars import *


def peak_frequency_pm(u10, g=9.81):
    """Return Pierson Moskovitz peak angular frequency σ.
    
    https://wikiwaves.org/Ocean-Wave_Spectra"""
    u195 = 1.026 * u10 
    σpm = (0.8777 * g) / u195
    return σpm


def spectrum_pm(f, u10=20, α=8.1e-3, β=0.74, g=9.81, angular=False):
    """Pierson and Moskovitz frequency spectrum.
    
    https://wikiwaves.org/Ocean-Wave_Spectra
    """
    const = 1 if angular else 2 * np.pi   
    ω = const * f
    u195 = 1.026 * u10 # assuming drag coefficient of 1.3e-3
    ω0 = g / u195
    S = ((α * g**2) / ω**5) * np.exp(-β * (ω0 / ω)**4)
    return S


def cos2_model(θ, θm):
    """Calculate directional spreading spectrum.
    
    §Holthuijsen (2007) (6.3.23) and Pierson et al. (1952).
    Tested: yes
    """
    θ = θ - θm
    D = np.zeros(θ.shape)
    D = np.where(abs(θ) <= (np.pi/2), (2/np.pi) * np.cos(θ)**2, D)
    return D