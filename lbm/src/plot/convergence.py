import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (FixedLocator, FixedFormatter, FuncFormatter,
                               LogLocator, NullFormatter)

from lbm.src.plot import style

def _sci(value, _pos=None):
    """Format as m x 10^e in mathtext, e.g. 9.2 x 10^-2."""
    if value <= 0:
        return ""
    exponent = int(np.floor(np.log10(value)))
    mantissa = value / 10.0**exponent
    return rf"${mantissa:.1f}\times10^{{{exponent}}}$"


def plot_convergence(Ny_values, L2_errors, save_path, title=None):
    """Log-log grid convergence with a fitted rate.

    Ticks are placed at the DATA, not on a generic decade grid: one x tick
    per resolution tested and one y tick per measured error, so the reader
    can read the slope off the axes directly rather than trusting the
    printed fit. Minor gridlines are kept because the uneven spacing within
    a decade is what makes the log scale legible as a log scale.

    Cyan marks what the solver produced, amber the fitted rate it is being
    measured against — the same roles those hues hold in the metric row.
    """
    if len(L2_errors) <= 1:
        return

    Ny_values = np.asarray(Ny_values, dtype=float)
    L2_errors = np.asarray(L2_errors, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))

    slope, intercept = np.polyfit(np.log(Ny_values), np.log(L2_errors), 1)
    fitted = np.exp(intercept) * Ny_values**slope

    # Fit underneath the data: the measurement is the subject, the fit is
    # the reference it is read against.
    colour, linestyle = style.S_FIT
    ax.plot(Ny_values, fitted, color=colour, ls=linestyle, lw=2,
            label=rf"Fitted rate: {slope:.2f}")

    colour, _ = style.S_NUM
    ax.plot(Ny_values, L2_errors, "o", color=colour, ms=8,
            markeredgecolor=style.GROUND, markeredgewidth=1.2,
            label=r"Measured $L_2$ error", zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")

    # One x tick per resolution actually tested.
    ax.xaxis.set_major_locator(FixedLocator(Ny_values))
    ax.xaxis.set_major_formatter(
        FixedFormatter([f"{int(n)}" for n in Ny_values]))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs="all", numticks=20))
    ax.xaxis.set_minor_formatter(NullFormatter())

    # One y tick per measured error, so each point is readable off the axis.
    ax.yaxis.set_major_locator(FixedLocator(L2_errors))
    ax.yaxis.set_major_formatter(FuncFormatter(_sci))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs="all", numticks=20))
    ax.yaxis.set_minor_formatter(NullFormatter())

    # Breathing room so the outermost points are not on the frame.
    ax.set_xlim(Ny_values.min() / 1.35, Ny_values.max() * 1.35)
    ax.set_ylim(L2_errors.min() / 1.6, L2_errors.max() * 1.6)

    ax.set_xlabel(r"Grid resolution $N_y$   (log scale)", fontsize=13)
    ax.set_ylabel(r"$L_2$ error   (log scale)", fontsize=13)
    ax.set_title(title or "Method convergence rate study", fontsize=16)

    style.apply_figure_style(fig, [ax])

    # Major grid ties each tick to its point; minor grid carries the decade
    # structure that identifies the axes as logarithmic.
    ax.grid(True, which="major", color=style.RULE, alpha=0.55, lw=0.8, ls="-")
    ax.grid(True, which="minor", color=style.RULE, alpha=0.20, lw=0.5, ls="-")
    ax.set_axisbelow(True)

    style.style_legend(ax.legend(fontsize=11, loc="best"))

    fig.tight_layout()
    style.save(fig, save_path, dpi=300)
    plt.close(fig)
