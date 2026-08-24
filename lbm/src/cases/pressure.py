from lbm.src.core.analytical import poiseuille_from_pressure
from lbm.src.core.lattice import BaseLattice
from lbm.src.core.kernels import macro_kernel


class PressurePoiseuille(BaseLattice):
    def __init__(self, ny=16, aspect=5, Re=1.0, tau_lbm=0.933, periodic_x=True, **kwargs):
        super().__init__(nx=ny*aspect, ny=ny, tau_lbm=tau_lbm, **kwargs)
        self.Re              = Re
        self.L_char          = ny - 2
        self.u_max           = Re * self.nu / self.L_char
        self.delta_rho       = 24 * self.nu * (self.nx - 1) * self.u_max / (self.L_char)**2
        self.rho_in          = 1.0 + self.delta_rho / 2
        self.rho_out         = 1.0 - self.delta_rho / 2
        self.obstacle[:, 0]  = True
        self.obstacle[:, -1] = True
        self.periodic_x      = periodic_x

    def macro(self):
        macro_kernel(self.f, self.ux, self.uy, self.rho, self.cx, self.cy, 0.0, self.obstacle)

    def apply_boundary_conditions(self):
        self.zou_he_pressure_west()
        self.zou_he_pressure_east()

    def analytical_profile(self):
        return poiseuille_from_pressure(self.nx, self.ny, self.delta_rho, self.nu)