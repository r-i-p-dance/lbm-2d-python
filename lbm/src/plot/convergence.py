import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (FixedLocator, FixedFormatter, FuncFormatter,
                               LogLocator, NullFormatter)

from lbm.src.plot import style


def plot_convergence(Ny_values, L2_errors, save_path, title=None,
                     modules=(11, 7)):
    """Log-log grid convergence with a fitted rate, sized to the poster grid.

    `modules` is the size of the PLOT RECTANGLE in grid units; the saved
    file is one module larger on every side. See style.poster_figure.

    Ticks are placed at the DATA, not on a generic decade grid: one x tick
    per resolution tested and one y tick per measured error, so the reader
    can read the slope off the axes directly rather than trusting the
    printed fit. Minor gridlines are kept because the uneven spacing within
    a decade is what makes the log scale legible as a log scale.

    There are no axis labels. One module of margin is 26.4 pt and the tick
    labels need all of it, so what the axes represent is said in the title.

    Cyan marks what the solver produced, amber the fitted rate it is being
    measured against — the same roles those hues hold in the metric row.
    """
    if len(L2_errors) <= 1:
        return

    Ny_values = np.asarray(Ny_values, dtype=float)
    L2_errors = np.asarray(L2_errors, dtype=float)

    fig, ax = style.poster_figure(*modules)

    slope, intercept = np.polyfit(np.log(Ny_values), np.log(L2_errors), 1)
    fitted = np.exp(intercept) * Ny_values**slope

    # Fit underneath the data: the measurement is the subject, the fit is
    # the reference it is read against.
    colour, linestyle = style.S_FIT
    ax.plot(Ny_values, fitted, color=colour, ls=linestyle,
            label=rf"fitted rate: {slope:.2f}")

    colour, _ = style.S_NUM
    ax.plot(Ny_values, L2_errors, "o", color=colour,
            markeredgecolor=style.GROUND, markeredgewidth=style.SPINE_PT,
            label=r"measured $L_2$ error", zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")

    # One x tick per resolution actually tested.
    ax.xaxis.set_major_locator(FixedLocator(Ny_values))
    ax.xaxis.set_major_formatter(
        FixedFormatter([f"{int(n)}" for n in Ny_values]))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs="all", numticks=20))
    ax.xaxis.set_minor_formatter(NullFormatter())

    # One y tick per measured error, so each point is readable off the axis,
    # divided by a common factor so the labels stay short — written out in
    # full these are 3.7x10^-3, 8.2x10^-4, 1.9x10^-4, 4.6x10^-5, which
    # rotated into the gutter are 47 pt tall and land 1.3 pt apart.
    exponent = style.factor_exponent(L2_errors)
    ax.yaxis.set_major_locator(FixedLocator(L2_errors))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _p=None: f"{v / 10.0**exponent:.2g}"))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs="all", numticks=20))
    ax.yaxis.set_minor_formatter(NullFormatter())

    # Breathing room so the outermost points are not on the frame.
    ax.set_xlim(Ny_values.min() / 1.35, Ny_values.max() * 1.35)
    ax.set_ylim(L2_errors.min() / 1.6, L2_errors.max() * 1.6)

    ax.set_title(title or "method convergence rate study")

    style.apply_figure_style(fig, [ax])

    # Major grid ties each tick to its point; minor grid carries the decade
    # structure that identifies the axes as logarithmic.
    ax.grid(True, which="major", color=style.RULE, alpha=style.GRID_ALPHA,
            lw=style.GRID_PT, ls="-")
    ax.grid(True, which="minor", color=style.RULE, alpha=style.GRID_MINOR_ALPHA,
            lw=style.GRID_PT / 2, ls="-")
    ax.set_axisbelow(True)

    # Pinned, not "best": error falls left to right, so the upper right is
    # always the empty corner. "best" would move the legend whenever the
    # study is re-run with another resolution.
    style.style_legend(ax.legend(loc="upper right"))

    style.rotate_y_labels(ax)
    style.factor_label(ax, style.factor_text(exponent))

    style.save_exact(fig, save_path)
    plt.close(fig)
