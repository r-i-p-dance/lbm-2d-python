import numpy as np
from lbm.src.cases.obstacle_base import ObstacleChannel


class Cylinder(ObstacleChannel):
    """Channel with a circular obstacle (staircase approximation)."""

    def __init__(self, diameter_fraction=0.2, cx_fraction=0.25,
                 cy_fraction=0.5, **kwargs):
        # Store fractions before super().__init__, which calls _add_obstacles
        self._diameter_fraction = diameter_fraction
        self._cx_fraction = cx_fraction
        self._cy_fraction = cy_fraction
        super().__init__(**kwargs)

        # Override L_char to use cylinder diameter instead of channel height
        D = self._diameter_fraction * (self.ny - 2)
        self.L_char = D
        self.u_max = self.Re * self.nu / self.L_char
        self._check_stability_at_init()

    def _add_obstacles(self):
        nx, ny = self.nx, self.ny
        H = ny - 2                                  # fluid channel height
        D = self._diameter_fraction * H             # diameter in cells
        R = D / 2.0                                 # radius
        cx = self._cx_fraction * nx                 # center x
        cy = self._cy_fraction * (ny - 1)           # center y (0.5 = centered)

        for i in range(nx):
            for j in range(ny):
                dist = np.sqrt((i - cx)**2 + (j - cy)**2)
                if dist <= R:
                    self.obstacle[i, j] = True