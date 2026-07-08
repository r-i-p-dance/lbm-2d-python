import numpy as np
from lbm.src.core.kernels import *

class Lattice:
    def __init__(self, nx=4, ny=16, tau_lbm=0.8, u_max=0.04):
        self.nx                  = nx
        self.ny                  = ny
        self.H_eff               = ny - 2
        self.it                  = 0
        self.tau_lbm             = tau_lbm
        self.nu                  = (self.tau_lbm - 0.5) / 3.0
        self.u_max               = u_max
        self.g_x                 = 8 * self.nu * self.u_max / self.H_eff**2
        self.c                   = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                                             [1, 1], [-1, 1], [-1, -1], [1, -1]])
        self.cx                  = self.c[:, 0].astype(np.int64)
        self.cy                  = self.c[:, 1].astype(np.int64)
        self.cu                  = np.zeros((9, nx, ny))
        self.w                   = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
        self.opposite            = [0, 3, 4, 1, 2, 7, 8, 5, 6]
        self.obstacle            = np.zeros((nx, ny)).astype(bool)
        self.obstacle[:, 0]      = True   # The entire bottom edge is a wall
        self.obstacle[:, -1]     = True  # The entire top edge is a wall
        self.f                   = self.w[:, np.newaxis, np.newaxis] * np.ones((9, nx, ny))
        self.f_new               = np.zeros((9, nx, ny))
        self.f_eq                = np.zeros((9, nx, ny))
        self.rho                 = np.ones((nx, ny))
        self.ux                  = np.zeros((nx, ny))
        self.uy                  = np.zeros((nx, ny))
        self.cu                  = np.zeros((9, nx, ny))

    def macro(self):
        macro_kernel(self.f, self.ux, self.uy, self.rho, self.cx, self.cy, self.g_x, self.obstacle)

    def equilibrium(self):
        equilibrium_kernel(self.f_eq, self.ux, self.uy, self.rho, self.w, self.cx, self.cy)

    def collision(self):
        collide_kernel(self.f, self.f_eq, self.ux, self.uy,
                   self.w, self.cx, self.cy,
                   self.tau_lbm, self.g_x, self.obstacle)

    def bounce_back_obstacle(self):
        bounce_back_kernel(self.f, self.opposite, self.obstacle)

    def stream(self):
        stream_kernel(self.f, self.f_new, self.cx, self.cy)
        self.f, self.f_new = self.f_new, self.f

    def step(self):
        self.macro()
        self.equilibrium()
        self.collision()
        self.bounce_back_obstacle()
        self.stream()

    def run(self, n_steps):
        """Advance exactly n_steps time steps."""
        for _ in range(n_steps):
            self.step()

    def converge(self, tol=1e-8, check_every=500, max_steps=100_000_000):
        """Run until velocity stops changing. Returns the number of steps taken."""
        old_ux = self.ux.copy()
        for step in range(1, max_steps + 1):
            self.step()
            if step % check_every == 0:
                change = np.max(np.abs(self.ux - old_ux))
                old_ux = self.ux.copy()
                if change < tol:
                    self.it = step
                    return step
        raise RuntimeError(f"No convergence after {max_steps} steps (last change: {change:.3e})")


    