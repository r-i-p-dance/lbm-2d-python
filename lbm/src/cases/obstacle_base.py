import numpy as np

from lbm.src.core.lattice import BaseLattice
class ObstacleChannel(BaseLattice):
    """Channel with walls top/bottom, velocity inlet west, outflow east,
    and an obstacle mask defined by the subclass."""
    def __init__(self, nx=None, ny=16, aspect=5, Re=10.0, tau_lbm=0.933):
        super().__init__(nx=ny*aspect, ny=ny, tau_lbm=tau_lbm)
        self.Re = Re
        self.L_char = ny - 2
        self.u_max = Re * self.nu / self.L_char
        self.periodic_x = False
        self.obstacle[:, 0]  = True      # bottom wall
        self.obstacle[:, -1] = True      # top wall
        self._add_obstacles()
        self._check_stability_at_init()
        self.inlet_profile = self._compute_parabolic_inlet()

    def _add_obstacles(self):
        pass                   

    def _compute_parabolic_inlet(self):
        """Poiseuille profile that satisfies no-slip at effective wall positions.

        Overridden by cases that have partial-height inlets (e.g. BackwardStep).
        """
        ny = self.ny
        j = np.arange(ny)
        y_bot, y_top = 0.5, ny - 1.5
        H = y_top - y_bot                        # = ny - 2
        u = 4.0 * self.u_max * (j - y_bot) * (y_top - j) / H**2
        u[0] = 0.0                                # solid nodes
        u[-1] = 0.0
        return u          # base: no interior obstacles

    def apply_boundary_conditions(self):
        self.zou_he_velocity_west(self.inlet_profile)
        self.outflow_east()

    def _check_stability_at_init(self):
            Ma = self.u_max / (1.0 / np.sqrt(3.0))
            if Ma > 0.1:
                raise ValueError(
                    f"Mach number {Ma:.3f} exceeds 0.1 — LBM low-Mach assumption violated. "
                    f"Decrease Re, decrease tau, or increase Ny (currently {self.tau_lbm}, u_max={self.u_max:.4f})."
                )
            if Ma > 0.05:
                import warnings
                warnings.warn(f"Mach number {Ma:.3f} above 0.05 — compressibility error may be visible.")
    
            if self.tau_lbm <= 0.55:
                raise ValueError(f"tau={self.tau_lbm} too close to 0.5; BGK unstable.")