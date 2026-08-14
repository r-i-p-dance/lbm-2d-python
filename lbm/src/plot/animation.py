
from matplotlib import pyplot as plt
from matplotlib.animation import PillowWriter
import numpy as np


class Recorder:
    def __init__(self, lattice, path, fps=15, dpi=100, every=100, cmap='RdBu_r', interpolation='none'):
        self.lattice = lattice
        self.every = every
        aspect = lattice.nx / lattice.ny
        self.fig, self.ax = plt.subplots(figsize=(6*aspect, 6))
        self.img = self.ax.imshow(
            np.zeros((lattice.nx, lattice.ny)).T,
            cmap=cmap,
            vmin=0.0, vmax=lattice.u_max,
            interpolation=interpolation,
        )
        self.ax.set_axis_off()
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        self.writer = PillowWriter(fps=fps)
        self.writer.setup(self.fig, path, dpi=dpi)

    def maybe_capture(self, step):
        if step % self.every == 0:
            vel = np.sqrt(self.lattice.ux**2 + self.lattice.uy**2)
            self.img.set_data(vel.T)
            self.writer.grab_frame()
            return True
        return False

    def close(self):
        self.writer.finish()

def record_development(case_class, nx, ny, tol, every=100, path=None, cmap='RdBu_r', interpolation='none', **case_kwargs):
    ltc = case_class(nx=nx, ny=ny, **case_kwargs)
    rec = Recorder(ltc, path=path, every=every, cmap=cmap, interpolation=interpolation)
    ltc.converge(tol=tol, recorder=rec)
    rec.close()

