import numpy as np

from lbm.src.core.lattice import BaseLattice

class ObstacleChannel(BaseLattice):
    """Channel with walls top/bottom, velocity inlet west, outflow east,
    and an obstacle mask defined by the subclass."""
    def __init__(self, nx=None, ny=16, aspect=5, Re=20.0, tau_lbm=0.933):
        super().__init__(nx=ny*aspect, ny=ny, tau_lbm=tau_lbm)
        self.Re = Re
        self.L_char = ny - 2
        self.u_max = Re * self.nu / self.L_char
        self.periodic_x = False
        self.obstacle[:, 0]  = True      # bottom wall
        self.obstacle[:, -1] = True      # top wall
        self.inlet_profile = np.zeros(ny)
        self.inlet_profile[1:-1] = self.u_max
        self._add_obstacles()            # hook — subclass fills this in

    def _add_obstacles(self):
        pass                             # base: no interior obstacles

    def apply_boundary_conditions(self):
        self.zou_he_velocity_west(self.inlet_profile)
        self.outflow_east()
