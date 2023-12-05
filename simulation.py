# from Perplexity AI

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define domain
L = 1.0  # length of square domain
N = 100  # number of grid points in each direction
dx = L / (N - 1)  # grid spacing

# Set up initial conditions
h = np.ones((N, N))  # water depth
u = np.zeros((N, N))  # x-velocity
v = np.zeros((N, N))  # y-velocity

# Define boundary conditions on left side of square
h[:, 0] = 2.0  # water depth
u[:, 0] = 0.0  # x-velocity
v[:, 0] = 0.0  # y-velocity

# Choose time step based on CFL condition
dt = dx / np.sqrt(9.81 * np.max(h))

# Define function to update plot at each time step
def update_plot(frame):
    global h, u, v
    h_star = h - dt / dx * (u[1:, :] - u[:-1, :] + v[:, 1:] - v[:, :-1])
    u_star = u - dt / dx * (u[1:, :] * (u[1:, :] - u[:-1, :]) + v[:, :-1] * (u[1:, :] - u[:-1, :]) + \
                            0.5 * 9.81 * (h[1:, :] + h[:-1, :]) - h_star[1:, :] - h_star[:-1, :])
    v_star = v - dt / dx * (u[:-1, :] * (v[:, 1:] - v[:, :-1]) + v[:, :-1] * (v[:, 1:] - v[:, :-1]) + \
                            0.5 * 9.81 * (h[:, :-1] + h[:, 1:]) - h_star[:, :-1] - h_star[:, 1:])
    h_new = h_star - dt / dx * (u_star[1:, :] - u_star[:-1, :] + v_star[:, 1:] - v_star[:, :-1])
    u_new = u_star - dt / dx * (u_star[1:, :] * (u_star[1:, :] - u_star[:-1, :]) + \
                                v_star[:, :-1] * (u_star[1:, :] - u_star[:-1, :]) + \
                                0.5 * 9.81 * (h_star[1:, :] + h_star[:-1, :]) - h_new[1:, :] - h_new[:-1, :])
    v_new = v_star - dt / dx * (u_star[:-1, :] * (v_star[:, 1:] - v_star[:, :-1]) + \
                                v_star[:, :-1] * (v_star[:, 1:] - v_star[:, :-1]) + \
                                0.5 * 9.81 * (h_star[:, :-1] + h_star[:, 1:]) - h_new[:, :-1] - h_new[:, 1:])
    h[:] = h_new[:]
    u[:] = u_new[:]
    v[:] = v_new[:]
    plt.clf()
    plt.imshow(h)
    plt.colorbar()

# Create animation using FuncAnimation class
ani = FuncAnimation(plt.gcf(), update_plot)

# Show animation
plt.show()