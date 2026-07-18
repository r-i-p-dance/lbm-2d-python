
from matplotlib import pyplot as plt
from matplotlib.animation import PillowWriter
import numpy as np


class Recorder:
    def __init__(self, lattice, path, fps=15, dpi=100, every=100):
        self.lattice = lattice
        self.every = every
        self.fig, self.ax = plt.subplots()
        self.img = self.ax.imshow(np.zeros((lattice.nx, lattice.ny)).T, 
                                    cmap='inferno', vmin=0, vmax=lattice.u_max)
        self.writer = PillowWriter(fps=fps)
        self.writer.setup(self.fig, path, dpi=dpi)

    def maybe_capture(self, step):
        if step % self.every == 0:
            vel = np.sqrt(self.lattice.ux**2 + self.lattice.uy**2)
            self.img.set_data(vel.T)
            self.ax.set_title(f"step {step}")
            self.writer.grab_frame()

    def close(self):
        self.writer.finish()