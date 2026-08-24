import numpy as np

from lbm.src.cases.obstacle_base import ObstacleChannel

class BackwardStep(ObstacleChannel):
    def __init__(self, step_height_fraction=0.5, step_length_fraction=0.25, **kwargs):
        self._step_h_frac = step_height_fraction
        self._step_l_frac = step_length_fraction
        super().__init__(**kwargs)

        # Re should be based on inlet height (= H - step_height)
        H = self.ny - 2
        h_step = int(self._step_h_frac * H)
        self.L_char = H - h_step                # inlet height
        self.u_max = self.Re * self.nu / self.L_char
        self._check_stability_at_init()

        # Recompute inlet with correct u_max and step geometry
        self.inlet_profile = self._compute_parabolic_inlet()

    def _compute_parabolic_inlet(self):
        """Parabolic profile on the upper fluid inlet only.

        The step occupies the lower rows; the inlet parabola spans from
        y = h_step + 0.5 (top of step effective wall) to y = ny - 1.5 (top wall).
        """
        ny = self.ny
        H_full = ny - 2
        h_step = int(self._step_h_frac * H_full)

        u = np.zeros(ny)
        j = np.arange(ny)
        y_bot = h_step + 0.5
        y_top = ny - 1.5
        H_inlet = y_top - y_bot                          # = ny - 2 - h_step

        # apply parabola only in the inlet fluid region
        fluid = (j >= h_step + 1) & (j <= ny - 2)
        u[fluid] = 4.0 * self.u_max * (j[fluid] - y_bot) * (y_top - j[fluid]) / H_inlet**2
        return u

    def _add_obstacles(self):
        nx, ny = self.nx, self.ny
        H = ny - 2
        h_step = int(self._step_h_frac * H)
        L_step = int(self._step_l_frac * nx)
        self.obstacle[:L_step, 1:1 + h_step] = True