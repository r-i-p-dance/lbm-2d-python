from lbm.src.core.analytical import poiseuille_from_pressure
from lbm.src.core.lattice import BaseLattice
from lbm.src.core.kernels import macro_kernel, collide_kernel


class PressurePoiseuille(BaseLattice):
    def __init__(self, nx=None, ny=16, aspect=5, tau_lbm=0.933, u_max=0.04):
        super().__init__(nx=ny*aspect, ny=ny, tau_lbm=tau_lbm)
        self.u_max           = u_max
        self.H_eff           = ny - 2
        self.delta_rho       = 24 * self.nu * (self.nx - 1) * self.u_max / (self.H_eff)**2
        self.rho_in          = 1.0 + self.delta_rho / 2
        self.rho_out         = 1.0 - self.delta_rho / 2
        self.obstacle[:, 0]  = True
        self.obstacle[:, -1] = True
        self.periodic_x      = False

    def macro(self):
        macro_kernel(self.f, self.ux, self.uy, self.rho, self.cx, self.cy, 0.0, self.obstacle)

    def apply_boundary_conditions(self):
        self.zou_he_west()
        self.zou_he_east()

    def zou_he_west(self):
        # Zou-He boundary condition at the west boundary (inlet)
        for j in range(1, self.ny - 1):
            f0 = self.f[0, 0, j]
            f2 = self.f[2, 0, j]
            f3 = self.f[3, 0, j]
            f4 = self.f[4, 0, j]
            f6 = self.f[6, 0, j]
            f7 = self.f[7, 0, j]

            rho_w = self.rho_in
            self.uy[0, j] = 0.0
            self.ux[0, j] = 1 - ((f0+f2+f4 + 2*(f3+f6+f7)) / rho_w)
            self.rho[0, j] = rho_w

            self.f[1, 0, j] = f3 + (2/3) * rho_w * self.ux[0, j]
            self.f[5, 0, j] = f7 - 0.5 * (f2-f4) + (1/6) * rho_w * self.ux[0, j]
            self.f[8, 0, j] = f6 + 0.5 * (f2-f4) + (1/6) * rho_w * self.ux[0, j]
        
        

    def zou_he_east(self):
        # Zou-He boundary condition at the east boundary (outlet)
        E = self.nx - 1
        for j in range(1, self.ny - 1):
            xn = self.nx - 1
            f0 = self.f[0, E, j]
            f1 = self.f[1, E, j]
            f2 = self.f[2, E, j]
            f4 = self.f[4, E, j]
            f5 = self.f[5, E, j]
            f8 = self.f[8, E, j]

            rho_w = self.rho_out
            self.uy[E, j] = 0.0
            self.ux[E, j] = -1.0 + (f0 + f2 + f4 + 2*(f1 + f5 + f8)) / rho_w
            self.rho[E, j] = rho_w
            
            self.f[3, E, j] = f1 - (2/3) * rho_w * self.ux[E, j]
            self.f[7, E, j] = f5 + 0.5*(f2 - f4) - (1/6) * rho_w * self.ux[E, j]
            self.f[6, E, j] = f8 - 0.5*(f2 - f4) - (1/6) * rho_w * self.ux[E, j]

    def analytical_profile(self):
        return poiseuille_from_pressure(self.nx, self.ny, self.delta_rho, self.nu)