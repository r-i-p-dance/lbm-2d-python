
from matplotlib import pyplot as plt
from matplotlib.animation import PillowWriter
import numpy as np


class Recorder:
    def __init__(self, lattice, path, fps=15, dpi=100, every=100, max_steps=None, vmax_factor=1.0, cmap='RdBu_r', interpolation='none', print_progress=False):
        self.lattice = lattice
        self.every = every
        self.max_steps = max_steps
        self.print_progress = print_progress
        aspect = lattice.nx / lattice.ny
        self.fig, self.ax = plt.subplots(figsize=(6*aspect, 6))
        self.img = self.ax.imshow(
            np.zeros((lattice.nx, lattice.ny)).T,
            cmap=cmap,
            vmin=0.0, 
            vmax=lattice.u_max * vmax_factor,
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

            if self.print_progress:
                if self.max_steps is None:
                    print(f"\rStep {step}: max velocity = {np.max(vel):.6f}", end="", flush=True)
                else:
                    print(f"\rStep {step}/{self.max_steps}: max velocity = {np.max(vel):.4f}", end="", flush=True)

            return True
        return False

    def close(self):
        self.writer.finish()

def record_development(case_class, nx, ny, tol=None, Re=10.0, tau_lbm=0.933, max_steps=None, every=100, path=None, cmap='RdBu_r', interpolation='none', vmax_factor=1.0, print_progress=False, **case_kwargs):

    if (tol is None) == (max_steps is None):
            raise ValueError("Provide exactly one of tol or max_steps.")

    ltc = case_class(nx=nx, ny=ny, Re=Re, tau_lbm=tau_lbm, **case_kwargs)
    rec = Recorder(ltc, path=path, every=every, max_steps=max_steps, cmap=cmap, interpolation=interpolation, vmax_factor=vmax_factor, print_progress=print_progress)

    print(f"\nRecording {case_class.__name__} at {nx}x{ny}, Re={ltc.Re}, u_max = {ltc.u_max:.4f}, " + 
          f"Ma={ltc.u_max * np.sqrt(3):.3f}, every {every} steps ", end="", flush=True)
    if max_steps is None:
        print(f"until convergence to tol={tol} ...")
    else:
        print(f"for total {max_steps} steps…")
    
    if max_steps is not None:
        ltc.run(max_steps, recorder=rec)
    else:
        ltc.converge(tol=tol, recorder=rec)
    
    print(f"\nRecording complete. Saving to {path} ...", end="", flush=True)

    rec.close()

    print(f"\rRecording complete. Saved to {path}")
    print()

