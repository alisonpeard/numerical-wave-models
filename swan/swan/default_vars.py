import numpy as np

# programming defaults
order = 'F'               # Fortran-style, column major: last axis (x) slowest

# global constants
g = 9.81                  # acceleration due to gravity [ms-2]
ρw = 1.03                 # average density of seawater [gcm-3]

# base variables to use
d = 10                    # water depth [m]
u = np.array([0., 0.])      # ambient current [ms-1]
λ = 80                    # wave length [m]
θ = np.pi / 4             # direction of wave (from East pointing)
k = 2 * np.pi / λ         # wave number / spatial frequency of wave [m-1]
σ_lims = (0.05, 0.4)      # limits for σ
θ_lims = (-np.pi, np.pi)  # limits for θ
σmin = 0.04               #* (2 * np.pi)
σmax = 0.25               #* (2 * np.pi) # should be 1*2π according to Holthuijsen (2007)
k = np.array([k * np.cos(θ), k * np.sin(θ)])