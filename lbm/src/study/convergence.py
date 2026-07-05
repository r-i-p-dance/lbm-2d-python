from dataclasses import dataclass
import numpy as np

from lbm.src.core import analytical
from lbm.src.core.lattice import Lattice

@dataclass
class ConvergenceResult:
    Ny: int
    L2_error: float
    u_numerical: np.ndarray
    u_analytical: np.ndarray
    iterations: int

def run_convergence_study(resolutions, tau, u_max):
    for  r in resolutions:
        ltc = Lattice(nx=8, ny=r, tau_lbm=tau, u_max=u_max)
        ltc.nu = (ltc.tau_lbm - 0.5) / 3.0
        ltc.g_x = 8 * ltc.nu * ltc.u_max / (ltc.H_eff)**2

        ltc.run(100000)

        u_numerical = ltc.ux[ltc.nx//2, 1:-1]
        u_analytical = analytical.poiseuille_profile(ltc.ny, ltc.g_x, ltc.nu)

        L2_error = np.sqrt(np.sum((u_numerical - u_analytical)**2) / np.sum(u_analytical**2))

        yield ConvergenceResult(Ny=ltc.ny, L2_error=L2_error, u_numerical=u_numerical, u_analytical=u_analytical, iterations=ltc.it)
    