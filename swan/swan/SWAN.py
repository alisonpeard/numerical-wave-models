import numpy as np

from .numerical import *
from .default_vars import *
from .initial_conditions import *
from .relations_in import *
from .relations_out import *
from .spectral import *
from .spatial import *
from .solvers import *


class SWAN:
    def __init__(self, bathy, u10, theta, dx, ny=10, nx=10, nsigma=4, ntheta=3,
                 M=4, σscale='log'):
        self.u10 = u10[:ny, :nx]
        self.theta = theta[:ny, :nx]
        self.ny, self.nx = self.u10.shape
        self.n = self.ny * self.nx
        self.ntheta = ntheta
        self.nsigma = nsigma
        self.dims = [self.ntheta, self.nsigma, self.ny, self.nx]
        self.M = M
        self.σscale = σscale
        self.dx, self.dy = dx, dx
        
        # set up bathymetry
        bathy = bathy[:ny, :nx]
        self.land = bathy < 0
        bathy[self.land] = 1e-5 # fix later
        self.bathy = bathy
        
        # initialise variables for numerical solver
        self.sweeps = [{'id': m} for m in range(M)]
        self.sweeps[0]['thetas'] = np.linspace(0, np.pi/2, ntheta)
        self.sweeps[1]['thetas'] = np.linspace(np.pi/2, np.pi, ntheta)
        self.sweeps[2]['thetas'] = np.linspace(-np.pi/2, -np.pi, ntheta)
        self.sweeps[3]['thetas'] = np.linspace(0, -np.pi/2, ntheta)
        
        self.sigmas, self.dsigma = frequency_grid(nsigma, scale=σscale)
        
        self.stepθ = 1
        self.stepσ = self.ntheta
        self.stepy = self.stepσ * self.nsigma
        self.stepx = self.stepy * ny
        
        # flag attributes
        self.ready_to_solve = False
        self.latest_params = None
        
        
    def setup_solution_matrices(self, dt=1):
        for sweep in self.sweeps:
            thetas = sweep['thetas']
            sweep['dtheta'] = abs(thetas[1] - thetas[0])

            # make meshgrids for all the inputs
            *_, H  = np.meshgrid(thetas, self.sigmas, sweep_flip(self.bathy, sweep['id']).ravel(order=order), indexing='ij')
            *_, θm  = np.meshgrid(thetas, self.sigmas, sweep_flip(self.theta, sweep['id']).ravel(order=order), indexing='ij')
            θ, Σ, U10  = np.meshgrid(thetas, self.sigmas, sweep_flip(self.u10, sweep['id']).ravel(order=order), indexing='ij')
            
            # initial energy distribution
            Ef = spectrum_pm(Σ, U10)
            fpeak = peak_frequency_pm(U10) / (2 * np.pi)
            D = cos2_model(θ, θm)
            E = Ef * D
            N = E / Σ

            # velocities
            K = inverse_dispersion_approx(Σ, H)
            c = phase_speed(K, Σ)
            cg = group_velocity(K, H, Σ)
            cx, cy = spatial_group_velocity(cg, θ)
            cθ = directional_velocity(θ, c, cg, self.stepx, self.stepy, self.dx, self.dy)

            # make sure directions flipped correctly
            cx, cy = flip_spatial_velocities(cx, cy, sweep['id'])
            cθ = flip_directional_velocities(cθ, sweep['id'])
            # cσ = ...

            # update matrices
            A_geo = bsbt_matrix(self.stepx, self.stepy, cx, cy, self.dx, self.dy)
            A_spectral = theta_block_matrix(cθ, self.stepθ)
            A = A_spectral + A_geo
            I = np.eye(self.ntheta * self.nsigma * self.ny * self.nx)
            Ainv = np.linalg.inv(I + dt * A)

            # add to sweep data
            sweep['cy'] = cy
            sweep['cx'] = cx
            sweep['cθ'] = cθ
            sweep['θ'], sweep['Σ'] = θ, Σ
            sweep['N0'] = N
            sweep['A'] = Ainv
            sweep['fpeak'] = fpeak
            
        self.dtheta = self.sweeps[0]['dtheta']
        self.Σ_full = np.concatenate([sweep['Σ'] for sweep in self.sweeps], axis=0)
        self.ready_to_solve = True
    
    
    def do_sweeps(self, T, dt=1):
        """Do T sweeps with timestep dt, if not already done."""
        if self.latest_params != (T, dt):
            if not self.ready_to_solve:
                self.setup_solution_matrices(dt)
            for sweep in self.sweeps:
                sweep['N'] = sweep['N0']
                sweep['N(t)'] = [sweep['N0']]
            for t in range(T):
                for sweep in self.sweeps:
                    N1 = sweep['A'] @ sweep['N'].ravel(order=order)
                    E1 = energy_density(N1, sweep['Σ']).reshape(self.dims, order=order)
                    E1 = eliminate_negative_energies(E1, sweep['dtheta'], self.ntheta)
                    N1 = action_density(E1, sweep['Σ']).ravel(order=order)
                    sweep['N'] = N1
                    sweep['N(t)'].append(N1)
            self.latest_params = (T, dt)
    

    def solve_for_action_density(self, T, dt=1):
        self.do_sweeps(T, dt)
        self.N = []
        for t in range(T):
            N1 = [sweep['N(t)'][t].reshape(self.dims, order=order) for sweep in self.sweeps]
            N1 = [sweep_flip(N, sweep_id) for sweep_id, N in enumerate(N1)]
            N1 = np.concatenate([N for N in N1], axis=0)
            self.N.append(N1)
            
    
    def solve_for_swh(self, T, dt=1):
        self.do_sweeps(T, dt)
        self.swh = []
        for t in range(T):
            N1 = [sweep['N(t)'][t].reshape(self.dims, order=order) for sweep in self.sweeps]
            N1 = [sweep_flip(N, sweep_id) for sweep_id, N in enumerate(N1)]
            N1 = np.concatenate([N for N in N1], axis=0)
            E1 = energy_density(N1, self.Σ_full)
            Hs1 = significant_wave_height(E1, self.dtheta, self.dsigma)
            self.swh.append(Hs1)


    # model check methods
    def check_sigma_instability(self):
        ratio = self.sigmas.max() / self.sigmas.min()
        print(f"Ratio is r = {ratio}. Instabilities may arise if r >> 1. " \
              "See SWAN Technical Documentation Cycle III v40.51 for details.")