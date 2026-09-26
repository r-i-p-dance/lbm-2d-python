"""Two resolutions and their difference, in either of two layouts.

draw_field paints one field into an axes and knows nothing about figures or
files. Two builders wrap it: plot_comparison_poster writes each panel as its
own PDF on the poster grid, plot_comparison_combined stacks all three in one
figure for the README. Pick the builder at the call site — neither has to be
edited to get the other.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from lbm.src.plot import style

# Panel order, fixed. Names are file suffixes; titles are for the README
# layout, which is the only one that labels its panels.
NAMES = ("reference", "coarse", "difference")
TITLES = ("reference", "coarse (upsampled)", "absolute difference")


def draw_field(ax, data, obstacle, cmap, vmax, gamma):
    """Paint one field into an existing axes. No sizing, no saving.

    Solid cells are painted style.SOLID. Without a mask the geometry renders
    at the ramp's zero end and cannot be told from quiescent fluid — and on
    the difference panel a solid cell would look identical to a cell where the
    two grids agree perfectly, which is the one distinction that panel exists
    to make.
    """
    cmap = cmap.copy()
    cmap.set_bad(style.SOLID)
    ax.imshow(np.where(obstacle, np.nan, data).T, cmap=cmap, origin="lower",
              norm=mcolors.PowerNorm(gamma=gamma, vmin=0.0,
                                     vmax=max(vmax, 1e-12)),
              interpolation="nearest")
    # Frame kept, ticks dropped. At high resolution the solid cells are a
    # pixel or two wide and the domain boundary all but disappears, so the
    # spines are what tell you where the field actually ends.
    ax.set_xticks([])
    ax.set_yticks([])


def _panels(ref, coarse, obstacle_ref, obstacle_coarse, cmap):
    """The three panels: (data, obstacle, cmap, vmax).

    Reference and coarse share a colour scale, so the two can be read against
    each other. The difference gets its own, 0 to its own max, on the cool arm
    of the palette: putting the discrepancy on the opposite side of the
    cyan/amber axis means it can never be mistaken for more flow.

    The two fields keep SEPARATE masks. The coarse run resolves the geometry
    on its own grid, so upsampling it back gives a blockier staircase than the
    reference has, and that difference is part of what the study is measuring.
    The difference panel masks the union — the cells where both runs have
    fluid to compare, which is the set the L2 norm is taken over.
    """
    diff = np.abs(ref - coarse)
    vmax = max(ref.max(), coarse.max())
    return [(ref, obstacle_ref, cmap, vmax),
            (coarse, obstacle_coarse, cmap, vmax),
            (diff, obstacle_ref | obstacle_coarse,
             style.SEQUENTIAL_DUAL, diff.max())]


def plot_comparison_poster(ref, coarse, obstacle_ref, obstacle_coarse, stem,
                           modules, field=style.FIELD_VELOCITY):
    """One PDF per panel, each sized to the poster grid.

    `stem` is a path without extension; the files written are
    <stem>_reference.pdf, <stem>_coarse.pdf and <stem>_difference.pdf.
    Separate files because they go on the poster separately — stacked in one
    figure their relative placement is fixed and the group cannot be gridded.

    `modules` sizes the PLOT RECTANGLE; the PDF is one module larger on every
    side, the same gutter every other poster figure has. The image fills the
    rectangle, so pick modules in the domain's own aspect — these cases are
    nx = 5*ny, so (20, 4) or (15, 3) — or the field comes out stretched.

    No colorbar: it would either eat into the rectangle or break the gutter.
    """
    cmap, gamma = field
    for name, (data, obstacle, panel_cmap, vmax) in zip(
            NAMES, _panels(ref, coarse, obstacle_ref, obstacle_coarse, cmap)):
        fig, ax = style.poster_figure(*modules)
        draw_field(ax, data, obstacle, panel_cmap, vmax, gamma)
        # imshow sets aspect "equal", which would letterbox the image inside
        # the rectangle and leave dead space where the grid expects field.
        ax.set_aspect("auto")
        style.apply_figure_style(fig, [ax], image_axes=[ax])
        style.save_exact(fig, f"{stem}_{name}.pdf")
        plt.close(fig)


def plot_comparison_combined(ref, coarse, obstacle_ref, obstacle_coarse, path,
                             title=None, field=style.FIELD_VELOCITY):
    """All three panels stacked in one figure, for the README.

    imshow keeps the data's aspect, so a panel's height follows from its
    width. Deriving the figure height from that rather than fixing it leaves
    no dead band above and below each panel — the same reasoning the
    optimization recorder uses to make its panels tile exactly.
    """
    cmap, gamma = field
    nx, ny = ref.shape

    LEFT, RIGHT, TOP, BOTTOM = 0.02, 0.98, 0.93, 0.02
    fig_w = 6 * nx / ny
    panel_w = (RIGHT - LEFT) * fig_w
    panel_h = panel_w * ny / nx
    fig_h = panel_h * (3 + 2 * style.GUTTER) / (TOP - BOTTOM)

    fig, axes = plt.subplots(3, 1, figsize=(fig_w, fig_h),
                             gridspec_kw={"hspace": style.GUTTER,
                                          "left": LEFT, "right": RIGHT,
                                          "top": TOP, "bottom": BOTTOM})

    for ax, subtitle, (data, obstacle, panel_cmap, vmax) in zip(
            axes, TITLES,
            _panels(ref, coarse, obstacle_ref, obstacle_coarse, cmap)):
        draw_field(ax, data, obstacle, panel_cmap, vmax, gamma)
        ax.set_title(subtitle)

    if title:
        fig.suptitle(title, color=style.TEXT)

    style.apply_figure_style(fig, list(axes), image_axes=tuple(axes))
    style.save(fig, path, dpi=300)
    plt.close(fig)
