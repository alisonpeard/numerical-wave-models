import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from .default_vars import *


def subset_heatmap(ji, x, stepx, stepy, nx, ny):
    """Display subet of a 2d array x around ravelled index ji."""
    xsteps = np.arange(-(nx * stepx), (nx * stepx) + 1, stepx)
    ysteps = np.arange(-(ny * stepy), (ny * stepy) + 1, stepy)
    X, Y = np.meshgrid(xsteps, ysteps)
    indices = X + Y
    indices += ji
    return x[indices]


def flattened_heatmap(X, dims, which='z', ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots()
    X = flattened_meshgrid(X, dims, which)
    im = ax.imshow(X)
    plt.colorbar(im)


def flattened_meshgrid(X, dims, which='z'):
    ny, nx, nz0, nz1 = dims
    X = X.reshape(dims, order=order)
    if which == 'z':
        X = np.moveaxis(X, [0, 1, 2, 3], [0, 2, 1, 3])
        X = X.reshape([ny * nz0, nx * nz1], order=order)
    elif which == 'x':
        X = np.moveaxis(X, [0, 1, 2, 3], [3, 1, 2, 0])
        X = X.reshape([nz1 * nx, nz0 * ny], order=order)
    elif which == 'y':
        X = np.moveaxis(X, [0, 1, 2, 3], [1, 2, 0, 3])
        X = X.reshape([nz0 * ny, nz1 * nx], order=order)
    else:
        raise ValueError("Argument 'which' must be one of ['y', 'x', 'z'].")
    return X
    
    
def polar_heatmap(r, theta, Z, cmap='YlOrRd', title=r"$F(f)\cdot D(f,\theta)$", scale=1,
                  rscale='log', rticks=[]):
    """Plot directional frequency spectrum on polar axes."""
    gs=gridspec.GridSpec(1,1)
    gs.update(wspace=0.205, hspace=0.105) 
    fig = plt.figure(figsize=(scale * 500/72.27, scale * 450/72.27))
    ax = fig.add_subplot(gs[0,0], projection='polar')
    im = ax.contourf(theta, r, Z, 20, cmap=cmap)
    ax.tick_params(colors='k', axis="y", which='both')
    ax.set_yscale(rscale)
    ax.set_yticks(rticks)
    #ax.set_yticks([0.1, 0.15, 0.2, 0.3])
    ax.set_title(title)
    plt.colorbar(im)