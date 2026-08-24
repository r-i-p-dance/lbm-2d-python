from lbm.src.core.analytical import poiseuille_from_force
from lbm.src.core.lattice import BaseLattice
from lbm.src.core.kernels import macro_kernel, collide_kernel, forcing_kernel


class ForcedPoiseuille(BaseLattice):
    def __init__(self, nx=8, ny=16, Re=1.0, tau_lbm=0.933):
        super().__init__(nx=nx, ny=ny, tau_lbm=tau_lbm)
        self.Re              = Re
        self.L_char          = ny - 2
        self.u_max           = Re * self.nu / self.L_char
        self.g_x             = 8 * self.nu * self.u_max / self.L_char**2
        self.obstacle[:, 0]  = True
        self.obstacle[:, -1] = True
        self._check_stability_at_init()
        

    def macro(self):
        macro_kernel(self.f, self.ux, self.uy, self.rho, self.cx, self.cy, self.g_x, self.obstacle)

    def apply_forcing(self):
        forcing_kernel(self.f, self.ux, self.uy, self.w, self.cx, self.cy, self.tau_lbm, self.g_x, self.obstacle)

    def analytical_profile(self):
        return poiseuille_from_force(self.ny, self.g_x, self.nu)