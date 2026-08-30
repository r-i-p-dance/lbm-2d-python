import numpy as np
import matplotlib.pyplot as plt

from lbm.src.plot import style


def plot_profile_comparison(u_numerical, u_analytical, Ny, save_path, title=None):
    """
    Plot the comparison between numerical and analytical velocity profiles.

    Cyan is the LBM result, amber the analytical profile it is measured
    against; the residual is magenta because it is neither of the two.

    Parameters:
    u_numerical (numpy.ndarray): The numerical velocity profile (1D array).
    u_analytical (numpy.ndarray): The analytical velocity profile (1D array).
    Ny (int): Number of grid points in the y-direction.
    save_path (str): Path to save the plot image.
    title (str): Optional figure-level title.
    """
    y = np.arange(1, Ny-1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6.5))

    # =========================================================================
    # Subplot 1: Velocity Profiles
    # =========================================================================
    color, ls = style.S_EXACT
    ax1.plot(y, u_analytical, color=color, ls=ls, lw=2, label='Analytical')
    color, _ = style.S_NUM
    ax1.plot(y, u_numerical, 'o', color=color, ms=6,
             markeredgecolor=style.GROUND, markeredgewidth=0.6,
             label='LBM', zorder=3)

    ax1.set_xlabel('y', fontsize=13)
    ax1.set_ylabel('Velocity', fontsize=13)
    ax1.set_title('Velocity profile comparison', fontsize=15)
    style.style_legend(ax1.legend(fontsize=11))
    ax1.set_box_aspect(1)  # Forces the plot area to be perfectly square

    # =========================================================================
    # Subplot 2: Residuals
    # =========================================================================
    color, ls = style.S_RESID
    ax2.plot(y, u_numerical - u_analytical, marker='o', color=color, ls=ls, ms=4, lw=1.6,
             markeredgecolor=style.GROUND, markeredgewidth=0.6)
    # Zero is the reference the residual is read against, so it gets the
    # same amber as the analytical curve next to it.
    # ax2.axhline(0.0, color=style.AMBER, ls='--', lw=1.0, alpha=0.7)

    ax2.set_xlabel('y', fontsize=13)
    ax2.set_ylabel('Error', fontsize=13)
    ax2.set_title('Residual (numerical - analytical)', fontsize=15)
    ax2.set_box_aspect(1)  # Forces the plot area to be perfectly square

    if title:
        fig.suptitle(title, fontsize=17, color=style.TEXT)

    style.apply_figure_style(fig, [ax1, ax2])

    fig.tight_layout()
    style.save(fig, save_path, dpi=300)
    plt.close(fig)
