import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from lbm.src.plot import style


def _finish(fig, ax, title, save_path):
    """Style, title and write one poster figure.

    The y axis always factors its exponent into an offset. A margin is
    26.4 pt and a label like "-0.00003" is far wider than that, whereas the
    factored "-3.0" is 20 pt and fits. Forcing it unconditionally rather
    than on a magnitude threshold keeps every figure's left margin the same
    width, which is the point of a grid.
    """
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)

    # A tick label has 24.4 pt to live in: one module of margin less the tick
    # pad. Matplotlib's default step of 2.5 forces two decimals ("0.25"), and
    # "0.00" is 25.2 pt — 0.8 pt off the page. Dropping 2.5 from the ladder
    # keeps the mantissa to one decimal, and four bins is as many as a small
    # panel can carry legibly anyway.
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))

    ax.set_title(title)
    style.apply_figure_style(fig, [ax])
    ax.set_axisbelow(True)
    style.factor_label(ax)
    style.save_exact(fig, save_path)
    plt.close(fig)


def plot_profile_comparison(u_numerical, u_analytical, Ny, save_path,
                            title=None, modules=(11, 7), mode="poster"):
    """Numerical velocity profile against the analytical one.

    Cyan is the LBM result, amber the analytical profile it is measured
    against. `modules` sizes the plot rectangle on the poster grid; the
    saved file is one module larger on every side.

    No axis labels: a margin is 26.4 pt and the tick labels need it, so the
    title says what the axes are.
    """
    y = np.arange(1, Ny - 1)
    fig, ax = style.poster_figure(*modules, mode=mode)

    colour, linestyle = style.S_EXACT
    ax.plot(y, u_analytical, color=colour, ls=linestyle, label="exact")
    colour, _ = style.S_NUM
    ax.plot(y, u_numerical, "o", color=colour,
            markeredgecolor=style.GROUND, markeredgewidth=style.SPINE_PT,
            label="LBM", zorder=3)

    style.style_legend(ax.legend(loc="lower center"))
    _finish(fig, ax, title or "numerical vs analytical", save_path)


def plot_profile_residual(u_numerical, u_analytical, Ny, save_path,
                          title=None, modules=(11, 7), mode="poster"):
    """Numerical minus analytical, as its own poster figure.

    Magenta because the residual is neither of the two series it is drawn
    from. Split out of the comparison plot so it can be placed anywhere on
    the poster — or left off it.
    """
    y = np.arange(1, Ny - 1)
    fig, ax = style.poster_figure(*modules, mode=mode)

    residual = u_numerical - u_analytical
    colour, linestyle = style.S_RESID
    ax.plot(y, residual, marker="o", color=colour, ls=linestyle,
            markeredgecolor=style.GROUND, markeredgewidth=style.SPINE_PT)

    # Zero is always on the axis. Not decoration: for this case the residual
    # is a near-constant offset — at Ny=16 it varies by 2e-15 about 2.8e-5 —
    # and left to itself ScalarFormatter subtracts that constant and plots
    # the remainder, so the panel shows floating-point noise magnified to
    # full height. Anchoring to zero shows the residual's actual size, and
    # makes the four resolutions comparable to each other.
    lo, hi = min(residual.min(), 0.0), max(residual.max(), 0.0)
    pad = 0.08 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)

    _finish(fig, ax, title or "residual", save_path)
