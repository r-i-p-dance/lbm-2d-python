from lbm.src.cases.obstacle_base import ObstacleChannel

class TwoRectangles(ObstacleChannel):
    def _add_obstacles(self):
        nx, ny = self.nx, self.ny
        w = nx // 10          # block width
        h = ny // 2           # half-height
        x_top = nx // 5       # top block near inlet
        x_bot = nx // 5 * 2   # bottom block downstream
        self.obstacle[x_top:x_top + w, ny - h:] = True
        self.obstacle[x_bot:x_bot + w, :h]      = True