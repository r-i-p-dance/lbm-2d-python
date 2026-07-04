import numpy as np

class lattice:
    def __init__(self):
        self.nx                  = 300
        self.ny                  = 100
        self.tau_lbm             = 1.0
        self.nu                  = (self.tau_lbm - 0.5) / 3.0
        self.u_max               = 0.4
        self.g_x                 = 8 * self.nu * self.u_max / self.ny**2
        self.Nt                  = int(5 * self.ny**2 / self.nu) + 2000
        self.c                   = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                                             [1, 1], [-1, 1], [-1, -1], [1, -1]])
        self.w                   = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
        self.opposite            = [0, 3, 4, 1, 2, 7, 8, 5, 6]
        self.grid                = np.zeros((self.nx, self.ny))
        self.obstacle            = np.zeros((self.nx, self.ny)).astype(bool)
        self.obstacle[:, 0]      = True   # The entire bottom edge is a wall
        self.obstacle[:, -1]     = True  # The entire top edge is a wall
        self.f                   = self.w[:, np.newaxis, np.newaxis] * np.ones((9, self.nx, self.ny))
        self.f_eq                = np.zeros((9, self.nx, self.ny))
        self.rho                 = np.ones((self.nx, self.ny))
        self.ux                  = np.zeros((self.nx, self.ny))
        self.uy                  = np.zeros((self.nx, self.ny))
        self.cu                  = np.zeros((9, self.nx, self.ny))

    def macro(self):
        self.rho = np.sum(self.f, axis=0)
        self.ux = (np.sum(self.f * self.c[:, 0, np.newaxis, np.newaxis], axis=0) + self.g_x/2) / self.rho
        self.uy = np.sum(self.f * self.c[:, 1, np.newaxis, np.newaxis], axis=0) / self.rho

    def equilibrium(self):
        self.cu = self.c[:, 0, np.newaxis, np.newaxis] * self.ux + self.c[:, 1, np.newaxis, np.newaxis] * self.uy
        u_sq = self.ux**2 + self.uy**2
        self.f_eq = self.w[:, np.newaxis, np.newaxis] * self.rho * (1 + 3*self.cu + 4.5*self.cu**2 - 1.5*u_sq)

    def collision(self):
        self.cu = self.c[:, 0, np.newaxis, np.newaxis] * self.ux + self.c[:, 1, np.newaxis, np.newaxis] * self.uy
        F = self.w[:, np.newaxis,  np.newaxis] * (3*(self.c[:, 0, np.newaxis, np.newaxis] - self.ux) + 9*self.cu * self.c[:, 0, np.newaxis, np.newaxis]) * self.g_x

        self.f = self.f - (1.0/self.tau_lbm)*(self.f - self.f_eq) + (1 - 1/(2*self.tau_lbm)) * F

    def bounce_back_obstacle(self):
        self.f[:, self.obstacle] = self.f[self.opposite][:, self.obstacle]

    def stream(self):
        for i in range(9):
            self.f[i] = np.roll(self.f[i], shift=(self.c[i, 0], self.c[i, 1]), axis=(0, 1))

    