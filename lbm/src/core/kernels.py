import numpy as np
from numba import njit

@njit(cache=True)
def collide_kernel(f, f_eq, ux, tau, obstacle):
    """BGK collision, skipping solid nodes."""
    nx, ny = ux.shape
    inv_tau = 1.0 / tau
    for i in range(nx):
        for j in range(ny):
            if obstacle[i, j]:
                continue
            for k in range(9):
                f[k, i, j] = f[k, i, j] - inv_tau * (f[k, i, j] - f_eq[k, i, j])

@njit(cache=True)
def forcing_kernel(f, ux, uy, w, cx, cy, tau, g_x, obstacle):
    """Guo forcing, skipping solid nodes."""
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
                f[k, i, j] += force_prefactor * Fk

@njit(cache=True)
def equilibrium_kernel(f_eq, ux, uy, rho, w, cx, cy):
    nx, ny = ux.shape
    for i in range(nx):
        for j in range(ny):
            uxij = ux[i, j]
            uyij = uy[i, j]
            usq = uxij * uxij + uyij * uyij
            rhoij = rho[i, j]
            for k in range(9):
                cuk = cx[k] * uxij + cy[k] * uyij
                f_eq[k, i, j] = w[k] * rhoij * (1.0 + 3.0 * cuk + 4.5 * cuk * cuk - 1.5 * usq)

@njit(cache=True)
def macro_kernel(f, ux, uy, rho, cx, cy, g_x, obstacle):
    nx, ny = ux.shape
    for i in range(nx):
        for j in range(ny):
            if obstacle[i, j]:
                continue
            r = 0.0
            mx = 0.0
            my = 0.0
            for k in range(9):
                fk = f[k, i, j]
                r += fk
                mx += fk * cx[k]
                my += fk * cy[k]
            rho[i, j] = r
            mx += 0.5 * g_x   # half-force correction in fluid only
            ux[i, j] = mx / r
            uy[i, j] = my / r

@njit(cache=True)
def stream_kernel(f, f_new, cx, cy):
    nx, ny = f.shape[1], f.shape[2]
    for k in range(9):
        dx = cx[k]
        dy = cy[k]
        for i in range(nx):
            i_src = (i - dx) % nx
            for j in range(ny):
                j_src = (j - dy) % ny
                f_new[k, i, j] = f[k, i_src, j_src]

@njit(cache=True)
def bounce_back_kernel(f, opposite, obstacle):
    nx, ny = f.shape[1], f.shape[2]
    temp = np.empty(9)
    for i in range(nx):
        for j in range(ny):
            if not obstacle[i, j]:
                continue
            for k in range(9):
                temp[k] = f[k, i, j]
            for k in range(9):
                f[k, i, j] = temp[opposite[k]]