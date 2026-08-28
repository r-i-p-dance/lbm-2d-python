import numpy as np
from lbm.src.core.kernels import *

class BaseLattice:
    def __init__(self, nx=4, ny=16, tau_lbm=0.933):
        self.nx         = nx
        self.ny         = ny
        self.it         = 0
        self.tau_lbm    = tau_lbm
        self.nu         = (self.tau_lbm - 0.5) / 3.0
        self.c          = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                                    [1, 1], [-1, 1], [-1, -1], [1, -1]])
        self.cx         = self.c[:, 0].astype(np.int64)
        self.cy         = self.c[:, 1].astype(np.int64)
        self.w          = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
        self.opposite   = [0, 3, 4, 1, 2, 7, 8, 5, 6]
        self.obstacle   = np.zeros((nx, ny)).astype(bool)
        self.f          = self.w[:, np.newaxis, np.newaxis] * np.ones((9, nx, ny))
        self.f_new      = np.zeros((9, nx, ny))
        self.f_eq       = np.zeros((9, nx, ny))
        self.rho        = np.ones((nx, ny))
        self.ux         = np.zeros((nx, ny))
        self.uy         = np.zeros((nx, ny))
        self.periodic_x = True

    def macro(self):
        macro_kernel(self.f, self.ux, self.uy, self.rho, self.cx, self.cy, 0.0, self.obstacle)

    def equilibrium(self):
        equilibrium_kernel(self.f_eq, self.ux, self.uy, self.rho, self.w, self.cx, self.cy)

    def collision(self):
        collide_kernel(self.f, self.f_eq, self.ux, self.tau_lbm, self.obstacle)

    def bounce_back_obstacle(self):
        bounce_back_kernel(self.f, self.opposite, self.obstacle)

    def stream(self):
        stream_kernel(self.f, self.f_new, self.cx, self.cy, self.periodic_x)
        self.f, self.f_new = self.f_new, self.f

    def step(self):
        self.macro()
        self.equilibrium()
        self.collision()
        self.bounce_back_obstacle()
        self.stream()
        self.apply_boundary_conditions()  # <- Hook
        self.apply_forcing()              # <- Hook

    def apply_boundary_conditions(self):pass
    def apply_forcing(self):pass
    def analytical_profile(self): pass

    def run(self, n_steps, recorder=None):
        """Advance exactly n_steps time steps."""
        for step in range(1, n_steps+1):
            self.step()
            if recorder is not None:
                recorder.maybe_capture(step)

    def converge(self, tol=1e-8, check_every=500, max_steps=1_000_000, recorder=None):
        """Run until velocity stops changing. Returns the number of steps taken."""
        old_f = self.f.copy()
        for step in range(1, max_steps + 1):
            self.step()

            if recorder is not None:
                recorder.maybe_capture(step)

            if step % check_every == 0:
                self.check_stability_running()

                change = np.max(np.abs(self.f - old_f))
                old_f = self.f.copy()
                if change < tol:
                    self.macro()
                    self.equilibrium()
                    self.it = step
                    return step
                
        raise RuntimeError(f"No convergence after {max_steps} steps (last change: {change:.3e})")

    def _check_stability_at_init(self):
        pass  # <- Hook for subclasses to check stability at init

    def check_stability_running(self):
        """Raise if the low-Mach assumption is violated.

        Overridden to a no-op in solvers whose 'velocity' is not a fluid
        velocity — the adjoint field is a sensitivity, so its moments carry
        no Mach constraint.
        """
        max_vel = np.max(np.sqrt(self.ux**2 + self.uy**2))
        if max_vel > (1.0 / np.sqrt(3.0)) * 0.1:
            Re = getattr(self, "Re", "n/a")
            u_max = getattr(self, "u_max", float("nan"))
            raise ValueError(
                f"Mach number {max_vel / (1.0 / np.sqrt(3.0)):.3f} exceeds 0.1 "
                f"— LBM low-Mach assumption violated. "
                f"Decrease Re, decrease tau, or increase ny. "
                f"(Currently Re={Re}, tau={self.tau_lbm}, u_max={u_max:.3f})")


    def zou_he_pressure_west(self):
        # Zou-He boundary condition at the west boundary (inlet)
        nb_zou_he_pressure_west(self.f, self.ux, self.uy, self.rho, self.rho_in)
            
    def zou_he_pressure_east(self, j_from, j_to):
        # Zou-He boundary condition at the east boundary (outlet)
        nb_zou_he_pressure_east(self.f, self.ux, self.uy, self.rho, self.rho_out, j_from, j_to)

    def zou_he_velocity_west(self, u_profile, j_from, j_to):
        # Zou-He boundary condition at the west boundary (inlet)
        nb_zou_he_velocity_west(self.f, self.ux, self.uy, self.rho, u_profile, j_from, j_to)

    def zou_he_velocity_south(self, u_profile, i_from, i_to):
        nb_zou_he_velocity_south(self.f, self.ux, self.uy, self.rho, u_profile, i_from, i_to)
        
    def outflow_east(self):
        # zero-gradient: copy second-to-last column into the last
        self.f[:, -1, :] = self.f[:, -2, :]