import numpy as np
from numba import njit

@njit(cache=True)
def collide_kernel(f, f_eq, ux, uy, w, cx, cy, tau, g_x, obstacle):
    """BGK collision with Guo forcing, skipping solid nodes."""
    nx, ny = ux.shape
    inv_tau = 1.0 / tau
    force_prefactor = 1.0 - 0.5 * inv_tau
    for i in range(nx):
        for j in range(ny):
            if obstacle[i, j]:
                continue
            uxij = ux[i, j]
            uyij = uy[i, j]
            for k in range(9):
                cuk = cx[k] * uxij + cy[k] * uyij
                Fk = w[k] * (3.0 * (cx[k] - uxij) + 9.0 * cuk * cx[k]) * g_x
                f[k, i, j] = f[k, i, j] - inv_tau * (f[k, i, j] - f_eq[k, i, j]) + force_prefactor * Fk