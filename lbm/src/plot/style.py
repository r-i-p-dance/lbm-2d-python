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
DASH PATTERNS, never from muting hue. Paired series are always separated by
linestyle as well as value, so they remain readable in greyscale and under
any CVD type. The metric series take this to its conclusion and drop hue
altogether — see S_METRIC below for why that is a design choice about
attention, not a retreat from the palette.

Set THEME = "light" for a warm-paper variant using the same hues darkened
for white stock.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Neue Haas Grotesk Display Pro",
                        "Neue Haas Grotesk Text Pro",
                        "Helvetica Neue", "Helvetica", "Arial",
                        ],
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
# Adjoint momentum. The cool half of the diverging scale, extended down
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
# Forward velocity. The warm half of the same scale. Sharing the umber,
# amber and pale-gold anchors with DIVERGING means the adjoint panel and
# the positive lobe of the sensitivity panel are literally the same colours.
SEQUENTIAL_WARM = mcolors.LinearSegmentedColormap.from_list(
    "arc_warm",
    [GROUND, "#2A1206", "#5C2A08", "#92400E",
     "#C77812", "#FBBF24", "#FFE9A8", "#FFFBF0"] if THEME == "dark" else
    ["#FFFBF0", "#FFE9A8", "#FBBF24", "#C77812",
     "#92400E", "#5C2A08", "#2A1206", GROUND],
    N=256)

# Physical flow uses the warm ramp; dual and error quantities use the cool
# arm. Named aliases so figure code states the ROLE, not the hue.
SEQUENTIAL_FLOW = SEQUENTIAL_WARM      # velocity, momentum
SEQUENTIAL_DUAL = SEQUENTIAL_COOL      # adjoint, residuals, differences

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
# (colour, linestyle) — never colour alone.
#
# METRIC SERIES ARE ACHROMATIC. These panels sit directly beneath the field
# maps, and chroma there is not free: a saturated line is exactly as loud as
# the physics next to it, so the metric row ends up competing with the
# simulation for attention instead of supporting it. Reserving colour for
# the fields is what keeps the eye going to them first.
#
# Separation instead comes from LUMINANCE plus DASH PATTERN — the same two
# mechanisms the CVD strategy above already relies on, with hue removed.
# That makes these the most robust series in the file: they survive
# greyscale, every CVD type, and a cheap poster print.
S_METRIC = (TEXT,  "-")      # the quantity a panel is about
S_ALT    = (MUTED, "--")     # its partner, where a panel shows two
S_REF    = (MUTED, ":")      # a limit or threshold — a rule, not data

# Verification series. These DO keep their chroma: they are standalone
# figures with no field map beside them to compete with, and the hue is
# carrying meaning — amber is the physical/reference quantity, cyan the
# computed result being checked against it, magenta the residual because
# it is neither.
S_NUM   = (CYAN,    "-")
S_EXACT = (AMBER,   "-")
S_FIT   = (AMBER,   "--")
S_RESID = (MAGENTA, "-")

# -------------------------------------------------------- solid material
# Obstacles and walls. Separated from the flow ramp by HUE rather than
# brightness: the ramp lives on the amber axis, so a cool near-black reads
# as a different substance rather than as "slightly more flow". Kept close
# to GROUND in value so the geometry is present without competing with the
# physics for attention.

# Tested:
# 1. #131A2B - too bright and distinct from the ground
# 2. #0B0D15 - too dark
# 3. #0D1019 - seems fine for now
# 4. #0E121D
# 5. #101422
# 6. #121724

SOLID = "#0D1019"



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
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.grid(color=RULE, alpha=0.45, linewidth=0.6)


def style_colorbar(cb):
    """Match a colorbar's frame and ticks to the figure rules."""
    cb.outline.set_edgecolor(RULE)
    cb.outline.set_linewidth(0.8)
    cb.ax.tick_params(colors=MUTED, labelcolor=MUTED, labelsize=7)
    cb.ax.yaxis.get_offset_text().set_color(MUTED)


def style_legend(leg):
    """Legends sit on the panel, not on a white card."""
    frame = leg.get_frame()
    frame.set_facecolor(PANEL)
    frame.set_edgecolor(RULE)
    frame.set_linewidth(0.8)
    for text in leg.get_texts():
        text.set_color(TEXT)
    return leg


def save(fig, path, dpi=300, **kwargs):
    """Save without matplotlib punching a white border round the ground."""
    kwargs.setdefault("bbox_inches", "tight")
    fig.savefig(path, dpi=dpi, facecolor=GROUND, edgecolor="none", **kwargs)