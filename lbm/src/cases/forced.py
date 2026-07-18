from lbm.src.core.lattice import BaseLattice
from lbm.src.core.kernels import macro_kernel, collide_kernel, forcing_kernel


class ForcedPoiseuille(BaseLattice):
    def __init__(self, nx=8, ny=16, tau_lbm=0.933, u_max=0.04):
        super().__init__(nx=nx, ny=ny, tau_lbm=tau_lbm)
        self.u_max           = u_max
        self.H_eff           = ny - 2
        self.g_x             = 8 * self.nu * self.u_max / self.H_eff**2
        self.obstacle[:, 0]  = True
        self.obstacle[:, -1] = True
        

    def macro(self):
        macro_kernel(self.f, self.ux, self.uy, self.rho, self.cx, self.cy, self.g_x, self.obstacle)

    def apply_forcing(self):
        forcing_kernel(self.f, self.ux, self.uy, self.w, self.cx, self.cy, self.tau_lbm, self.g_x, self.obstacle)