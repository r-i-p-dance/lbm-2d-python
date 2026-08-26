"""Visual identity for all project figures.

Design rationale
----------------
GROUND. Near-black. On white, every colour is a step DOWN in luminance and
reads as pigment; on near-black every accent is a step UP and reads as
emissive. This is what lets saturated colour stay saturated instead of
looking like ink on paper. It also suits a Swiss/Neue Haas Grotesk layout,
where the subject is figure-ground contrast and grid.

HUE AXIS. Cyan <-> amber. This is the one diverging axis that stays
separable under deuteranopia, protanopia AND tritanopia (red-green does
not), and it happens to be among the highest-chroma pairs that also survive
print. Safety and vibrancy are not in conflict here.

CVD STRATEGY. Keep chroma high; get separability from luminance spacing and
DASH PATTERNS, never from muting hue. Paired series (alpha/beta, fwd/adj)
are always colour + linestyle, so they remain readable in greyscale and
under any CVD type.

FIELD MAPS. inferno for sequential data: perceptually uniform, CVD-safe,
and its dark end merges into the ground so the data appears to emerge from
the page rather than sitting in a box.

Set THEME = "light" for a warm-paper variant using the same hues darkened
for white stock.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Neue Haas Grotesk Display Pro",
                        "Neue Haas Grotesk Text Pro",
                        "Helvetica Neue", "Helvetica", "Arial",
                        "DejaVu Sans"],
    "font.weight": "medium",
    "axes.titleweight": "medium",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Neue Haas Grotesk Display Pro",
    "mathtext.it": "Neue Haas Grotesk Display Pro:italic",
    "mathtext.bf": "Neue Haas Grotesk Display Pro:bold",
})

THEME = "dark"           # "dark" | "light"

# ---------------------------------------------------------------- dark
_DARK = dict(
    GROUND="#08090F",       # figure background — near-black, blue-shifted
    PANEL="#0E1018",        # panel background, one step up
    RULE="#2A2E3D",         # gridlines, spines
    TEXT="#EEF0F7",         # primary text
    MUTED="#8A90A6",        # tick labels, secondary text
    CYAN="#22D3EE",
    AMBER="#FBBF24",
    MAGENTA="#F0398B",
    VIOLET="#A78BFA",
    MINT="#34D399",
    CORAL="#FB7185",
)

# --------------------------------------------------------------- light
# Same hues, darkened for luminance contrast against warm paper.
_LIGHT = dict(
    GROUND="#F7F5F0",
    PANEL="#FFFFFF",
    RULE="#D5D1C8",
    TEXT="#12141C",
    MUTED="#6B7185",
    CYAN="#0891B2",
    AMBER="#B45309",
    MAGENTA="#BE185D",
    VIOLET="#6D28D9",
    MINT="#047857",
    CORAL="#BE123C",
)

_P = _DARK if THEME == "dark" else _LIGHT
GROUND, PANEL, RULE = _P["GROUND"], _P["PANEL"], _P["RULE"]
TEXT, MUTED = _P["TEXT"], _P["MUTED"]
CYAN, AMBER = _P["CYAN"], _P["AMBER"]
MAGENTA, VIOLET = _P["MAGENTA"], _P["VIOLET"]
MINT, CORAL = _P["MINT"], _P["CORAL"]

# ------------------------------------------------- sequential: cool arm
# Forward velocity. The cool half of the diverging scale, extended down
# into the ground and up into ice-white. Anchors at 0.24, 0.62 and 0.85
# luminance are the same three cyans used in DIVERGING, so the panels read
# as one system rather than three unrelated maps.
#
# Luminance rises monotonically across every anchor — that is what makes
# magnitude legible as brightness, and it is the property that breaks if
# you interpolate between saturated cyan and saturated amber directly
# (RGB interpolation routes that path through green).
SEQUENTIAL_COOL = mcolors.LinearSegmentedColormap.from_list(
    "arc_cool",
    [GROUND, "#0B1E3D", "#0D4A6E", "#0E7490",
     "#17A8C9", "#22D3EE", "#7DF9FF", "#E8FEFF"] if THEME == "dark" else
    ["#E8FEFF", "#7DF9FF", "#22D3EE", "#17A8C9",
     "#0E7490", "#0D4A6E", "#0B1E3D", GROUND],
    N=256)

# ------------------------------------------------- sequential: warm arm
# Adjoint momentum. The warm half of the same scale. Sharing the umber,
# amber and pale-gold anchors with DIVERGING means the adjoint panel and
# the positive lobe of the sensitivity panel are literally the same colours.
SEQUENTIAL_WARM = mcolors.LinearSegmentedColormap.from_list(
    "arc_warm",
    [GROUND, "#2A1206", "#5C2A08", "#92400E",
     "#C77812", "#FBBF24", "#FFE9A8", "#FFFBF0"] if THEME == "dark" else
    ["#FFFBF0", "#FFE9A8", "#FBBF24", "#C77812",
     "#92400E", "#5C2A08", "#2A1206", GROUND],
    N=256)

# Kept for anything that wants a single default sequential map.
SEQUENTIAL = SEQUENTIAL_COOL

# --------------------------------------------------- diverging: fields
# Centre is the GROUND colour, not white: on a dark figure a white centre
# glares and becomes the loudest thing on the panel, when zero sensitivity
# should be the quietest. Cyan for negative, amber for positive — the one
# diverging axis separable under all three CVD types.
DIVERGING = mcolors.LinearSegmentedColormap.from_list(
    "cyan_amber",
    ["#7DF9FF", "#22D3EE", "#0E7490", GROUND,
     "#92400E", "#FBBF24", "#FFE9A8"] if THEME == "dark" else
    ["#0E7490", "#22D3EE", "#A5F3FC", GROUND,
     "#FDE68A", "#F59E0B", "#B45309"],
    N=256)

# -------------------------------------------------------- design field
# Deliberately near-neutral with a cool cast: the design panel is the
# ANSWER, so it should read as form rather than compete with the two data
# fields for chroma. Solid merges into the ground; fluid is a bright
# channel carved out of the block.
DESIGN = mcolors.LinearSegmentedColormap.from_list(
    "design",
    [GROUND, "#141C2E", "#2E3C55", "#5E7091",
     "#9DAFC9", "#DCE6F5"] if THEME == "dark" else
    ["#12141C", "#2E3646", "#5E6878", "#98A2B2",
     "#CDD4DE", GROUND],
    N=256)

# ------------------------------------------------------------- series
# (colour, linestyle) — never colour alone. Paired series use the two
# identity hues so the metric row belongs to the same palette as the
# fields; the rest fall back to the secondary accents.
S_LOSS  = (CYAN,    "-")
S_LAM   = (MAGENTA, "-")
S_ALPHA = (AMBER,   "-")
S_BETA  = (VIOLET,  "--")
S_FWD   = (CYAN,    "-")
S_ADJ   = (AMBER,   "--")


def apply_figure_style(fig, axes, image_axes=()):
    """Apply the ground, rules and type colours to a figure."""
    fig.patch.set_facecolor(GROUND)
    for ax in axes:
        ax.set_facecolor(PANEL if ax not in image_axes else GROUND)
        for s in ax.spines.values():
            s.set_color(RULE)
            s.set_linewidth(0.8)
        ax.tick_params(colors=MUTED, labelcolor=MUTED)
        ax.title.set_color(TEXT)
        ax.grid(color=RULE, alpha=0.45, linewidth=0.6)