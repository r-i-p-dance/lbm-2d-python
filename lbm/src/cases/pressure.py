from lbm.src.core.analytical import poiseuille_from_pressure
from lbm.src.core.lattice import BaseLattice
from lbm.src.core.kernels import macro_kernel


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
        self.zou_he_pressure_west()
        self.zou_he_pressure_east()

    def analytical_profile(self):
        return poiseuille_from_pressure(self.nx, self.ny, self.delta_rho, self.nu)