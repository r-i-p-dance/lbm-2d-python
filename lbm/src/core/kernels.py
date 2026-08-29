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
def stream_kernel(f, f_new, cx, cy, periodic_x):
    nx, ny = f.shape[1], f.shape[2]
    for k in range(9):
        dx = cx[k]
        dy = cy[k]
        for i in range(nx):
            for j in range(ny):
                j_src = (j - dy) % ny
                if periodic_x:
                    i_src = (i - dx) % nx
                    f_new[k, i, j] = f[k, i_src, j_src]
                else:
                    i_src = i - dx
                    if 0 <= i_src < nx:
                        f_new[k, i, j] = f[k, i_src, j_src]
                    else:
                        f_new[k, i, j] = f[k, i, j]

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

@njit(cache=True)
def nb_zou_he_pressure_west(f, ux, uy, rho, rho_in):
    _, ny = ux.shape
    for j in range(1, ny - 1):
        f0 = f[0, 0, j]
        f2 = f[2, 0, j]
        f3 = f[3, 0, j]
        f4 = f[4, 0, j]
        f6 = f[6, 0, j]
        f7 = f[7, 0, j]

        uy[0, j] = 0.0
        ux[0, j] = 1 - ((f0 + f2 + f4 + 2 * (f3 + f6 + f7)) / rho_in)
        rho[0, j] = rho_in

        f[1, 0, j] = f3 + (2 / 3) * rho_in * ux[0, j]
        f[5, 0, j] = f7 - 0.5 * (f2 - f4) + (1 / 6) * rho_in * ux[0, j]
        f[8, 0, j] = f6 + 0.5 * (f2 - f4) + (1 / 6) * rho_in * ux[0, j]

@njit(cache=True)
def nb_zou_he_pressure_east(f, ux, uy, rho, rho_out, j_from, j_to):
    nx,_   = ux.shape
    E      = nx - 1
    for j in range(j_from, j_to):
        f0 = f[0, E, j]
        f1 = f[1, E, j]
        f2 = f[2, E, j]
        f4 = f[4, E, j]
        f5 = f[5, E, j]
        f8 = f[8, E, j]

        uy[E, j] = 0.0
        ux[E, j] = -1.0 + (f0 + f2 + f4 + 2 * (f1 + f5 + f8)) / rho_out
        rho[E, j] = rho_out

        f[3, E, j] = f1 - (2 / 3) * rho_out * ux[E, j]
        f[7, E, j] = f5 + 0.5 * (f2 - f4) - (1 / 6) * rho_out * ux[E, j]
        f[6, E, j] = f8 - 0.5 * (f2 - f4) - (1 / 6) * rho_out * ux[E, j]

@njit(cache=True)
def nb_zou_he_pressure_south(f, ux, uy, rho, rho_out, i_from, i_to):
    """Zou-He pressure BC on the south boundary (y=0), u_x = 0.
    Applied only on columns [i_from, i_to) — the outlet opening."""
    for i in range(i_from, i_to):
        f0 = f[0, i, 0]
        f1 = f[1, i, 0]
        f3 = f[3, i, 0]
        f4 = f[4, i, 0]
        f7 = f[7, i, 0]
        f8 = f[8, i, 0]

        ux[i, 0] = 0.0
        uy[i, 0] = 1.0 - ((f0 + f1 + f3 + 2 * (f4 + f7 + f8)) / rho_out)
        rho[i, 0] = rho_out

        f[2, i, 0] = f4 + (2 / 3) * rho_out * uy[i, 0]
        f[5, i, 0] = f7 - 0.5 * (f1 - f3) + (1 / 6) * rho_out * uy[i, 0]
        f[6, i, 0] = f8 + 0.5 * (f1 - f3) + (1 / 6) * rho_out * uy[i, 0]

@njit(cache=True)
def nb_zou_he_velocity_west(f, ux, uy, rho, u_profile, j_from, j_to):
    """Zou-He velocity BC on the west boundary, applied only on rows
    [j_from, j_to) — the inlet opening. u_profile is indexed from j_from."""
    for j in range(j_from, j_to):
        f0 = f[0, 0, j]; f2 = f[2, 0, j]; f3 = f[3, 0, j]
        f4 = f[4, 0, j]; f6 = f[6, 0, j]; f7 = f[7, 0, j]

        ux_j = u_profile[j - j_from]
        rho_j = (f0 + f2 + f4 + 2 * (f3 + f6 + f7)) / (1.0 - ux_j)
        ux[0, j] = ux_j
        uy[0, j] = 0.0
        rho[0, j] = rho_j

        f[1, 0, j] = f3 + (2 / 3) * rho_j * ux_j
        f[5, 0, j] = f7 - 0.5 * (f2 - f4) + (1 / 6) * rho_j * ux_j
        f[8, 0, j] = f6 + 0.5 * (f2 - f4) + (1 / 6) * rho_j * ux_j

@njit(cache=True)
def nb_zou_he_velocity_south(f, ux, uy, rho, u_profile, i_from, i_to):
    """Zou-He velocity BC on the south boundary (y=0), u_x = 0. u_y is prescribed."""

    for i in range(i_from, i_to):
        f0 = f[0, i, 0]
        f1 = f[1, i, 0]
        f3 = f[3, i, 0]
        f4 = f[4, i, 0]
        f7 = f[7, i, 0]
        f8 = f[8, i, 0]

        uy_i = u_profile[i - i_from]
        rho_i = (f0 + f1 + f3 + 2.0 * (f4 + f7 + f8)) / (1.0 - uy_i)

        ux[i, 0] = 0.0
        uy[i, 0] = uy_i
        rho[i, 0] = rho_i

        f[2, i, 0] = f4 + (2.0 / 3.0) * rho_i * uy_i
        f[5, i, 0] = f7 - 0.5 * (f1 - f3) + (1.0 / 6.0) * rho_i * uy_i
        f[6, i, 0] = f8 + 0.5 * (f1 - f3) + (1.0 / 6.0) * rho_i * uy_i

@njit(cache=True)
def nb_zou_he_velocity_east(f, ux, uy, rho, u_profile, j_from, j_to):
    """Zou-He velocity BC on the east boundary (x=nx-1), u_y = 0."""
    nx = ux.shape[0]
    east = nx - 1
    for j in range(j_from, j_to):
        f0 = f[0, east, j]
        f1 = f[1, east, j]
        f2 = f[2, east, j]
        f4 = f[4, east, j]
        f5 = f[5, east, j]
        f8 = f[8, east, j]

        ux_j = u_profile[j - j_from]
        rho_j = (f0 + f2 + f4 + 2.0 * (f1 + f5 + f8)) / (1.0 + ux_j)

        ux[east, j] = ux_j
        uy[east, j] = 0.0
        rho[east, j] = rho_j

        f[3, east, j] = f1 - (2.0 / 3.0) * rho_j * ux_j
        f[7, east, j] = f5 + 0.5 * (f2 - f4) - (1.0 / 6.0) * rho_j * ux_j
        f[6, east, j] = f8 - 0.5 * (f2 - f4) - (1.0 / 6.0) * rho_j * ux_j


@njit(cache=True)
def brinkman_collide_kernel(f, f_eq, omega_eff, obstacle):
    """BGK collision with a per-cell relaxation rate, skipping solids."""
    nx, ny = obstacle.shape
    for i in range(nx):
        for j in range(ny):
            if obstacle[i, j]:
                continue
            w = omega_eff[i, j]
            for k in range(9):
                f[k, i, j] -= w * (f[k, i, j] - f_eq[k, i, j])


@njit(cache=True)
def adjoint_collide_kernel(f, ux, uy, omega_eff, source, w, cx, cy, obstacle):
    """Adjoint collision: build the transposed-Jacobian equilibrium from the
    moments A, Bx, By and relax toward it, adding the source.

    Fused into one kernel — the moments are per-cell, so there is no reason
    to materialise E, D_x, D_y as (9,nx,ny) arrays as the NumPy version does.
    """
    nx, ny = obstacle.shape
    for i in range(nx):
        for j in range(ny):
            if obstacle[i, j]:
                continue

            u_x = ux[i, j]
            u_y = uy[i, j]
            usq = u_x * u_x + u_y * u_y

            A = 0.0
            Bx = 0.0
            By = 0.0
            for k in range(9):
                cu = cx[k] * u_x + cy[k] * u_y
                E = 1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq
                Dx = 3.0 * cx[k] + 9.0 * cu * cx[k] - 3.0 * u_x
                Dy = 3.0 * cy[k] + 9.0 * cu * cy[k] - 3.0 * u_y
                fk = f[k, i, j]
                A += w[k] * E * fk
                Bx += w[k] * Dx * fk
                By += w[k] * Dy * fk

            om = omega_eff[i, j]
            for k in range(9):
                f_eq_k = A + Bx * (cx[k] - u_x) + By * (cy[k] - u_y)
                f[k, i, j] += -om * (f[k, i, j] - f_eq_k) + source[k, i, j]