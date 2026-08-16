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
        self._check_stability()

    def _add_obstacles(self):
        nx, ny = self.nx, self.ny
        H = ny - 2
        h_step = int(self._step_h_frac * H)
        L_step = int(self._step_l_frac * nx)
        self.obstacle[:L_step, 1:1 + h_step] = True