import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from lbm.src.plot import style


def plot_field_comparison(u_ref_norm, u_up_norm, save_path,
                          title=None, cmap_field=None, cmap_diff=None,
                          gamma=0.8, obstacle_ref=None, obstacle_coarse=None,
                          obstacle_diff=None):
    """Reference, upsampled coarse, and their absolute difference side-by-side.

    Left and middle share a colormap scaled to reference u_max.
    Right uses its own scale from 0 to max difference — telling us where and
    how much the two disagree.

    The two fields use the warm flow ramp the animations use; the
    difference panel uses the cool arm. Putting the discrepancy on the
    opposite side of the cyan/amber axis means it can never be mistaken for
    more flow, and the panel stays readable under any CVD type.

    Solid cells are painted style.SOLID, exactly as the animations paint
    them. Without a mask the geometry renders at the ramp's zero end and
    cannot be told from quiescent fluid — and on the difference panel a
    solid cell would look identical to a cell where the two grids agree
    perfectly, which is the one distinction that panel exists to make.

    obstacle_ref / obstacle_coarse: the two fields need SEPARATE masks. The
    coarse run resolves the geometry on its own grid, so upsampling it back
    gives a blockier staircase than the reference has. That difference is
    part of what the study is measuring, and a single shared mask would
    paint over it.

    obstacle_diff: mask for the difference panel. Defaults to the union of
    the other two — the cells where both runs have fluid to compare, which
    is the same set the L2 norm is taken over.
    """
    diff = np.abs(u_ref_norm - u_up_norm)
    nx, ny = u_ref_norm.shape

    if cmap_field is None:
        cmap_field = style.SEQUENTIAL_FLOW
    if cmap_diff is None:
        cmap_diff = style.SEQUENTIAL_COOL

    if obstacle_diff is None and obstacle_ref is not None \
            and obstacle_coarse is not None:
        obstacle_diff = (np.asarray(obstacle_ref, dtype=bool)
                         | np.asarray(obstacle_coarse, dtype=bool))

    def masked(field, obstacle, cmap):
        """Hide the solid cells from both the colour scale and the panel."""
        if obstacle is None:
            return field, cmap
        cmap = cmap.copy()
        cmap.set_bad(style.SOLID)
        return np.where(np.asarray(obstacle, dtype=bool), np.nan, field), cmap

    ref_data, ref_cmap = masked(u_ref_norm, obstacle_ref, cmap_field)
    up_data, up_cmap = masked(u_up_norm, obstacle_coarse, cmap_field)
    diff_data, diff_cmap = masked(diff, obstacle_diff, cmap_diff)

    # nanmax, not max: masked cells are NaN, and the colour scale has to
    # come from the fluid alone.
    field_vmax = float(max(np.nanmax(ref_data), np.nanmax(up_data)))
    diff_vmax = float(np.nanmax(diff_data))

    aspect = nx / ny
    fig, axes = plt.subplots(3, 1, figsize=(6 * aspect, 12))

    panels = [
        (axes[0], ref_data,  'Reference (Ny=ref)',      ref_cmap,  field_vmax),
        (axes[1], up_data,   'Coarse (upsampled)',      up_cmap,   field_vmax),
        (axes[2], diff_data, 'Absolute difference',     diff_cmap, diff_vmax),
    ]

    for ax, data, subtitle, cmap, vmax in panels:
        img = ax.imshow(data.T, cmap=cmap, origin='lower',
                        norm=mcolors.PowerNorm(gamma=gamma, vmin=0.0,
                                               vmax=max(vmax, 1e-12)),
                        interpolation='nearest')
        ax.set_axis_off()
        ax.set_title(subtitle, fontsize=10, color=style.TEXT)
        cb = fig.colorbar(img, ax=ax, orientation='vertical',
                          pad=0.01, fraction=0.03)
        style.style_colorbar(cb)

    if title:
        fig.suptitle(title, fontsize=15, y=0.995, color=style.TEXT)

    style.apply_figure_style(fig, list(axes), image_axes=tuple(axes))

    fig.tight_layout()
    style.save(fig, save_path, dpi=100)
    plt.close(fig)
