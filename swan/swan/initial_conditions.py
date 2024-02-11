import numpy as np
from numba import njit
from numpy.linalg import norm as l2
from .default_vars import *


def peak_frequency_pm(u10, g=g):
    """Return Pierson Moskovitz peak angular frequency σ.
    
    https://wikiwaves.org/Ocean-Wave_Spectra"""
    u195 = 1.026 * u10 
    σpm = (0.8777 * g) / u195
    return σpm


def significant_wave_pm(u10, g=g):
    """https://wikiwaves.org/Ocean-Wave_Spectra"""
    return 0.22 * (u10)**2 / g


def spectrum_pm(f, u10=20, α=8.1e-3, β=0.74, g=g, angular=False):
    """Pierson and Moskovitz frequency spectrum.
    
    https://wikiwaves.org/Ocean-Wave_Spectra
    """
    const = 1 if angular else 2 * np.pi   
    ω = const * f
    u195 = 1.026 * u10 # assuming drag coefficient of 1.3e-3
    ω0 = g / u195
    S = ((α * g**2) / ω**5) * np.exp(-β * (ω0 / ω)**4)
    return S


def peak_enhancement(f, u10, F, g=g, angular=True):
    """https://wikiwaves.org/Ocean-Wave_Spectra"""
    const = 1 if angular else 2 * np.pi   
    ω = const * f
    ωp = 22 * (g**2 / (u10 * F))**(1/3)
    γ = 3.3
    σ = 0.07 if ω <= ωp else 0.09
    r = np.exp(- (ω - ωp)**2 / (2 * σ**2 * ω**2))
    return γ**r


def spectrum_JONSWAP(f, u10, F, angular=True):
    α = 0.076 * (u10**2 / (F * g))**(0.22)
    γr = peak_enhancement(f, u10, F)
    S = spectrum_pm(f, u10, α, angular=angular) * γr
    return S

def significant_wave_height_JONSWAP(u10, F, g=g):
    m0 = 1.67e-7 * F * (u10)**2 / g
    return 4 * np.sqrt(m0)


def cos2_model(θ, θm):
    """Calculate directional spreading spectrum.
    
    §Holthuijsen (2007) (6.3.23) and Pierson et al. (1952).
    Tested: yes
    """
    θ = θ - θm
    D = np.zeros(θ.shape)
    D = np.where(abs(θ) <= (np.pi/2), (2/np.pi) * np.cos(θ)**2, D)
    return D