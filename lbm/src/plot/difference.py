import matplotlib.pyplot as plt
import numpy as np

def plot_field_comparison(u_ref_norm, u_up_norm, save_path,
                          title=None, cmap_field='viridis', cmap_diff='inferno'):
    """Reference, upsampled coarse, and their absolute difference side-by-side.

    Left and middle share a colormap scaled to reference u_max.
    Right uses its own scale from 0 to max difference — telling us where and
    how much the two disagree.
    """
    diff = np.abs(u_ref_norm - u_up_norm)
    nx, ny = u_ref_norm.shape
    field_vmax = max(u_ref_norm.max(), u_up_norm.max())

    aspect = nx / ny
    fig, axes = plt.subplots(3, 1, figsize=(6 * aspect, 12))

    panels = [
        (axes[0], u_ref_norm,    'Reference (Ny=ref)',      cmap_field, field_vmax),
        (axes[1], u_up_norm,     'Coarse (upsampled)',      cmap_field, field_vmax),
        (axes[2], diff,          'Absolute difference',     cmap_diff,  diff.max()),
    ]

    for ax, data, subtitle, cmap, vmax in panels:
        img = ax.imshow(data.T, cmap=cmap, vmin=0.0, vmax=vmax,
                        interpolation='nearest', origin='lower')
        ax.set_axis_off()
        ax.set_title(subtitle, fontsize=10)
        fig.colorbar(img, ax=ax, orientation='vertical',
                     pad=0.01, fraction=0.03)

    if title:
        fig.suptitle(title, fontsize=15, y=0.995)

    fig.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)