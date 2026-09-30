"""Two resolutions and their difference.

draw_field paints one field into an axes and knows nothing about figures or
files. Two builders wrap it: plot_comparison_poster writes each panel as its
own PDF, for placing on the poster by hand; plot_comparison_combined puts all
three in one figure, for a README. Both land on the module grid.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from lbm.src.plot import style

# Panel order, fixed. These are the file suffixes the poster builder writes;
# the combined layout titles its panels from the resolutions instead.
NAMES = ("reference", "coarse", "difference")


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
                           modules, field=style.FIELD_VELOCITY,
                           mode="poster"):
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
        fig, ax = style.poster_figure(*modules, mode=mode)
        draw_field(ax, data, obstacle, panel_cmap, vmax, gamma)
        # imshow sets aspect "equal", which would letterbox the image inside
        # the rectangle and leave dead space where the grid expects field.
        ax.set_aspect("auto")
        style.apply_figure_style(fig, [ax], image_axes=[ax])
        style.save_exact(fig, f"{stem}_{name}.pdf")
        plt.close(fig)


def _cells(square, w, h, m):
    """Where the three panels sit, in modules from the bottom-left.

    One module between panels in both arrangements; `m` to every edge.
    `w, h` is the DIFFERENCE panel — the one the figure is really about, `m`
    the outer margin — and the layout follows from the domain's shape:

      pyramid   a square domain. Three of those in a column is a tall ribbon,
                so the two resolutions go small and side by side above the
                difference. They are (w - 1) / 2 square, so the pair plus the
                gap between them spans the difference exactly — the relation
                the vertical recorder uses. w must be ODD.

      stacked   anything wider. These channels are 5:1; side by side they
                would be unreadable, so a column is the only arrangement that
                works, and all three stay the same size.
    """
    if square:
        s = (w - 1) // 2
        return ([(m, m + h + 1, s, s), (m + s + 1, m + h + 1, s, s),
                 (m, m, w, h)],
                (w + 2 * m, h + s + 1 + 2 * m))
    return ([(m, m + 2 * (h + 1), w, h), (m, m + h + 1, w, h), (m, m, w, h)],
            (w + 2 * m, 3 * h + 2 + 2 * m))


def plot_comparison_combined(ref, coarse, obstacle_ref,
                             obstacle_coarse, path,
                             modules, ny_coarse, note=None,
                             field=style.FIELD_VELOCITY, mode="poster"):
    """All three panels in one figure on the module grid, for a README.

    `modules` sizes the DIFFERENCE panel, in the domain's own aspect; the rest
    of the layout follows from it — see _cells. The arrangement is read off
    the domain rather than passed in, because it only ever restates the shape
    of the data: square fields get the pyramid, wider ones the column.

    `ny_coarse` names the coarse grid in the titles, so the figure says what
    it is comparing without the reader going to the filename for it. Only the
    coarse one is asked for: `coarse` arrives already upsampled onto the
    reference grid — that is what makes the subtraction possible — so its own
    resolution is not in the array any more, while the reference's is.

    `note` is appended to the difference panel's title, which is where a
    study's number belongs — it is the difference that the L2 error or the
    agreement percentage describes.
    """
    cmap, gamma = field
    nx, ny = ref.shape
    cells, (total_w, total_h) = _cells(nx == ny, *modules,
                                       style.MARGINS[mode])
    titles = [rf"coarse, $N_y$ = {ny_coarse}",
              rf"reference, $N_y$ = {ny}",
              "absolute difference" + (f", {note}" if note else "")]

    unit = style.BASELINE_MM / style.MM_PER_IN
    fig = plt.figure(figsize=(total_w * unit, total_h * unit))

    axes = []
    for (x, y, w, h), title, (data, obstacle, panel_cmap, vmax) in zip(
            cells, titles,
            _panels(coarse, ref, obstacle_ref, obstacle_coarse, cmap)):
        ax = fig.add_axes([x / total_w, y / total_h, w / total_w, h / total_h])
        draw_field(ax, data, obstacle, panel_cmap, vmax, gamma)
        # imshow sets aspect "equal", which would letterbox the image inside
        # the rectangle and leave dead space where the grid expects field.
        ax.set_aspect("auto")
        ax.set_title(title)
        axes.append(ax)

    style.apply_figure_style(fig, axes, image_axes=axes)
    style.save_exact(fig, path)
    plt.close(fig)
