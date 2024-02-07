"""Solution algorithms for numerical methods."""
import numpy as np
from numba import njit
from scipy.linalg import block_diag
from .default_vars import *

# utils
def is_singular(A):
    det = np.linalg.det(A)
    return True if det == 0 else False

# exact methods
def exact_solution(A, b):
    """Exact solution, for comparison."""
    x = np.linalg.inv(A) @ b
    return x


# iterative methods
@njit
def gauss_seidel(A, b, x0, tol=1e-3, N=1000):
    """Burden and Faires Section 7.3.
    
    In the absence of source/sink terms: $(I + Δt*A)N^(n) = N^(n-1)$
    Note: assert not swan.is_singular(A) # do before function call"""
    n = len(b)
    L = -np.tril(A, -1)
    U = -np.triu(A, 1)
    D = np.diag(np.diag(A))
    DLinv = np.linalg.inv(D - L)
    Tg = DLinv @ U
    cg = DLinv @ b
    
    x = x0.copy()
    for k in range(N):
        x = Tg @ x + cg
        if (np.linalg.norm(x - x0, np.inf) / np.linalg.norm(x, np.inf)) < tol:
            return x
        x0 = x.copy()
    print(f"Gauss-Seidel: maximum number of iterations exceeded.")
    return x


# def gauss_seidel_iterative(A, b, x0, tol=1e-3, N=1000):
#     """Burden and Faires Section 7.3.
    
#     Iterative solution not working as accurately as martix, will return to later."""
#     n = len(b)
#     x = np.zeros(n)
#     for k in range(N):
#         for i in range(n):
#             x[i] -= (A[i, :i] * x[:i]).sum()
#             x[i] -= (A[i, i + 1:] * x0[i + 1:]).sum()
#             x[i] += b[i]
#             x[i] *= (1 / A[i, i])
#         if np.linalg.norm(x - x0, np.inf) < tol:
#             return x
#         x0 = x
#     print(f"Gauss-Seidel: maximum number of iterations exceeded.")
#     return x