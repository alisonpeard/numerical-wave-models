import numpy as np
# from numba import njit
from scipy.linalg import block_diag
from .default_vars import *
from .numerical import *

# geographic space
def bsbt_matrix(stepx, stepy, cx, cy, dx, dy):
    """Construct BSBT scheme matrix.
    
    Tested: yes
    """
    cx = cx.ravel(order=order)
    cy = cy.ravel(order=order)
    A = np.diag((cx / dx) + (cy / dy))
    A -= np.diag((cx[stepx:] / dx), -stepx)
    A -= np.diag((cy[stepy:] / dy), -stepy)
    return A


def sordup_matrix(nx, ny, cx, cy, dx, dy):
    """Construct SORDUP scheme matrix.
    
    Tested: no
    """
    A = np.diag(3 * (cx.ravel() / (2 * dx)) + 3 * (cy.ravel() / (2 * dy)))
    # x updates
    A -= np.diag(4 * (cx.ravel()[1:] / (2 * dx)), -1)
    A += np.diag((cx.ravel()[2:] / (2 * dx)), -2)
    # y updates
    A -= np.diag(4 * (cy.ravel()[nx:] / (2 * dy)), -nx)
    A += np.diag((cy.ravel()[2 * nx:] / (2 * dy)), -2 * nx)
    return A