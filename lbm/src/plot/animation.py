
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import PillowWriter
import numpy as np

from lbm.src.plot import style


class Recorder:
    """Records velocity magnitude as a GIF on the project ground.

    The field is drawn full-bleed: no axes, no frame, no padding. On a
    poster or a README the panel is the figure, and a white matplotlib
    border round a near-black field is the one thing that would break it.

    Solid cells are masked out rather than left at zero. Zero velocity and
    solid would otherwise be the same colour, so the cylinder and the step
    would dissolve into the quiescent fluid around them; painting them
    style.SOLID separates them from the flow ramp by hue rather than
    brightness, so they read as a different substance instead of as
    slightly more flow.
    """

    def __init__(self, lattice, path, fps=15, dpi=100,
                 max_steps=None,
                 every=None, every_min=None, every_max=None, accelerate_over=None,
                 vmax_factor=None, cmap=None, interpolation='none',
                 gamma=0.8,
                 print_progress=False):

        constant_mode = every is not None
        adaptive_args = (every_min, every_max, accelerate_over)
        adaptive_mode = all(a is not None for a in adaptive_args)
        if constant_mode == adaptive_mode:                       # both or neither
            raise ValueError(
                "Provide either `every` (constant) or all of "
                "`every_min`, `every_max`, `accelerate_over` (adaptive) — not both."
            )

        # Store schedule parameters
        if constant_mode:
            self._every_min = self._every_max = every
            self._decay_rate = 0.0                               # no growth
        else:
            self._every_min = every_min
            self._every_max = every_max
            # Choose decay so period reaches ~95% of every_max at accelerate_over
            self._decay_rate = 3.0 / accelerate_over

        self.lattice = lattice
        self.max_steps = max_steps
        self.print_progress = print_progress

        self._next_capture = 0

        # Solid cells never move, so the mask is fixed for the whole run.
        self._solid = np.asarray(lattice.obstacle, dtype=bool)

        if cmap is None:
            cmap = style.SEQUENTIAL_FLOW
        cmap = (cmap if isinstance(cmap, mcolors.Colormap)
                else plt.get_cmap(cmap)).copy()
        cmap.set_bad(style.SOLID)

        # Figure setup
        aspect = lattice.nx / lattice.ny
        self.fig, self.ax = plt.subplots(figsize=(6*aspect, 6))
        self.fig.patch.set_facecolor(style.GROUND)
        self.ax.set_facecolor(style.GROUND)
        self.img = self.ax.imshow(
            np.zeros((lattice.nx, lattice.ny)).T,
            cmap=cmap,
            # gamma < 1 lifts the low-velocity half of the range, so the
            # field is gold where there is flow rather than navy everywhere
            # but the jet core.
            norm=mcolors.PowerNorm(gamma=gamma, vmin=0.0,
                                   vmax=lattice.u_max * vmax_factor),
            interpolation=interpolation,
            origin='lower'
        )
        self.ax.set_axis_off()
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        self.writer = PillowWriter(fps=fps)
        self.writer.setup(self.fig, path, dpi=dpi)

    def _period_at(self, step):
        """Recording period at this step: constant if decay_rate=0, else growing."""
        return int(self._every_max
                   - (self._every_max - self._every_min) * np.exp(-self._decay_rate * step))

    def maybe_capture(self, step):
        if step < self._next_capture:
            return False

        vel = np.sqrt(self.lattice.ux**2 + self.lattice.uy**2)
        self.img.set_data(np.where(self._solid, np.nan, vel).T)
        self.writer.grab_frame(facecolor=style.GROUND, edgecolor="none")

        self._next_capture = step + self._period_at(step)

        if self.print_progress:
            suffix = f"/{self.max_steps}" if self.max_steps else ""
            print(f"\rStep {step}{suffix}: max |u| = {np.max(vel):.4f}",
                  end="", flush=True)
        return True

    def save_last_frame(self, path, dpi=300):
        """Still of the final state, for the poster and the README."""
        self.fig.savefig(path, dpi=dpi, facecolor=style.GROUND,
                         edgecolor="none", bbox_inches="tight", pad_inches=0)
        return path

    def close(self):
        self.writer.finish()
        plt.close(self.fig)

def record_development(case_class, nx, ny, path=None, Re=10.0, tau_lbm=0.933,
                       tol=None, max_steps=None,
                       every=None, every_min=None, every_max=None, accelerate_over=None,
                       cmap=None, interpolation='none',
                       vmax_factor=None,
                       gamma=0.8, still_path=None,
                       print_progress=False,
                       **case_kwargs):

    if (tol is None) == (max_steps is None):
            raise ValueError("Provide exactly one of tol or max_steps.")

    ltc = case_class(nx=nx, ny=ny, Re=Re, tau_lbm=tau_lbm, **case_kwargs)
    if vmax_factor is None:
        print(f"\nvmax_factor not specified. Warming up to estimate max velocity for {case_class.__name__} at {nx}x{ny}, Re={ltc.Re}, tau={ltc.tau_lbm}…", end="", flush=True)

        # Warmup to get a good estimate of max velocity
        ltc.run(3000, recorder=None)
        vmax = np.max(np.sqrt(ltc.ux**2 + ltc.uy**2))
        vmax_factor = vmax / ltc.u_max

        print(f"\rvmax_factor not specified. Estimated max velocity: {vmax:.4f}. Setting vmax_factor={vmax_factor:.3f} for recording.")

        ltc = case_class(nx=nx, ny=ny, Re=Re, tau_lbm=tau_lbm, **case_kwargs)

    rec = Recorder(
        ltc, path=path,
        every=every, every_min=every_min, every_max=every_max, accelerate_over=accelerate_over,
        max_steps=max_steps,
        cmap=cmap, interpolation=interpolation, vmax_factor=vmax_factor,
        gamma=gamma,
        print_progress=print_progress
    )

    ma = ltc.u_max * np.sqrt(3)
    mode = f"until convergence to tol={tol} ..." if tol is not None else f"for total {max_steps} steps…"
    print(f"\nRecording {case_class.__name__} at {nx}x{ny}, "
          f"Re={ltc.Re}, u_max={ltc.u_max:.4f}, Ma={ma:.3f}, {mode}")

    if max_steps is not None:
        ltc.run(max_steps, recorder=rec)
    else:
        ltc.converge(tol=tol, recorder=rec)

    print(f"\nRecording complete. Saving to {path} ...", end="", flush=True)

    if still_path is not None:
        rec.save_last_frame(still_path)

    rec.close()

    print(f"\rRecording complete. Saved to {path}")
    if still_path is not None:
        print(f"Final frame saved to {still_path}")
    print()
