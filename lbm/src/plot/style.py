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

import math

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Poster grid. Every figure dimension derives from these, so a change to
# the grid propagates to every figure rather than being re-entered by hand.
BASELINE_MM = 9.31          # one leading unit, 26.4 pt
MM_PER_IN = 25.4
MODULE_PT = BASELINE_MM / MM_PER_IN * 72     # one module, in points

# Type sizes in POINTS — absolute, never scaled by figure size. That is the
# whole point: a 6-module figure and a 20-module one carry identical type, so
# they can sit side by side on the poster and read as one system.
PLOT_TITLE_PT  = 13
AXIS_LABEL_PT  = 13
TICK_LABEL_PT  = 13
LEGEND_PT      = 13

# Stroke weights in POINTS, absolute for the same reason.
LINE_PT        = 1.4
MARKER_PT      = 5.0
SPINE_PT       = 0.8
GRID_PT        = 0.5
TICK_LEN_PT    = 3.0
# Grid opacity. Minor rules carry the decade structure on a log axis and must
# stay clearly subordinate to the major ones, or the panel reads as ruled
# paper. Here rather than at each call site so the two never drift apart.
GRID_ALPHA       = 0.55
GRID_MINOR_ALPHA = 0.20
# Ticks point inward, so a label needs less clearance than the default
# 3.5 pt assumes. The saving matters: a margin is 26.4 pt and "-7.5"
# at 13 pt is 27.3 pt, which overflows at the default pad and fits here.
TICK_PAD_PT    = 2.0

# Set once here so no figure has to repeat a size at a call site. Anything
# passing an explicit fontsize= or lw= is opting out of the system and will
# drift from the rest of the poster.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Neue Haas Grotesk Display Pro",
                        "Neue Haas Grotesk Text Pro",
                        "Helvetica Neue", "Helvetica", "Arial",
                        ],
    # regular, not medium: only Light/Bold/Black of Neue Haas are installed,
    # so asking for medium silently embeds a face nobody chose.
    "font.weight": "regular",
    "axes.titleweight": "regular",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Neue Haas Grotesk Display Pro",
    "mathtext.it": "Neue Haas Grotesk Display Pro:italic",
    "mathtext.bf": "Neue Haas Grotesk Display Pro:bold",

    "axes.titlesize": PLOT_TITLE_PT,
    # Pin the title to the top of the axes, which also switches OFF
    # matplotlib's auto-placement. That placement inspects the y axis offset
    # text and nudges the title clear of it, so titles ended up at different
    # heights from panel to panel — and where the offset is hidden but still
    # carries a string, it pushed the title to infinity and it vanished. On a
    # fixed grid every title belongs at the same height by construction.
    "axes.titley": 1.0,
    "axes.labelsize": AXIS_LABEL_PT,
    "xtick.labelsize": TICK_LABEL_PT,
    "ytick.labelsize": TICK_LABEL_PT,
    "legend.fontsize": LEGEND_PT,
    "lines.linewidth": LINE_PT,
    "lines.markersize": MARKER_PT,
    "axes.linewidth": SPINE_PT,
    "grid.linewidth": GRID_PT,
    # Inward ticks: the gutter is one leading and the tick labels need all
    # of it, so the marks cannot also live out there.
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": TICK_LEN_PT,
    "ytick.major.size": TICK_LEN_PT,
    "xtick.minor.size": TICK_LEN_PT / 2,
    "ytick.minor.size": TICK_LEN_PT / 2,
    "xtick.major.pad": TICK_PAD_PT,
    "ytick.major.pad": TICK_PAD_PT,
    # TrueType, not matplotlib's default Type 3 subsets: Type 3 text is not
    # selectable in InDesign and prints unpredictably.
    "pdf.fonttype": 42,
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

# --------------------------------------------------------- field roles
# (ramp, gamma). A figure names the ROLE and takes both from here, so no
# call site carries a colormap or a bare exponent.
#
# gamma < 1 lifts the low-velocity half of the range, so a flow field is gold
# where there is flow rather than navy everywhere but the jet core. rho_bar
# already spans [0, 1], so the design field needs no lift.
FIELD_VELOCITY = (SEQUENTIAL_FLOW, 0.8)
FIELD_DESIGN   = (DESIGN, 1.0)

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
# Obstacles and walls. Now simply PANEL: one fewer colour in the palette,
# and solid material reads at the same value as the metric panels beside it.
# Still separated from the flow ramp by HUE rather than brightness — the
# ramp lives on the amber axis, so a cool near-black reads as a different
# substance rather than as "slightly more flow" — and still close enough to
# GROUND that the geometry is present without competing with the physics.

# Tested:
# 1. #131A2B - too bright and distinct from the ground
# 2. #0B0D15 - too dark
# 3. #0D1019 - seems fine for now
# 4. #0E121D
# 5. #101422
# 6. #121724
# 7. PANEL (#0E1018) - current; a shade of its own was never doing much
#                      work, and reusing PANEL is one less thing to tune

# Swap the alias below for any hex above to go back.
SOLID = PANEL



# Outer margin, in modules, per destination. On the poster the page sits in
# the layout's own one-leading gutter, so one is enough and anything more
# double-counts it. A README figure has nothing around it but the page, so it
# carries its own breathing room. Every figure builder in the project takes a
# `mode` and looks the margin up here, so the two destinations stay
# consistent across the recorder, the fields and the line plots.
MARGINS = {"poster": 1, "readme": 2}


def poster_figure(w_modules, h_modules, mode="poster"):
    """A figure whose AXES RECTANGLE is exactly w x h grid modules.

    The usual matplotlib flow is backwards for a poster: you give a figure
    size, the layout engine decides how much of it the axes gets, and the
    plot rectangle ends up whatever is left over after the labels. Here the
    caller sizes the rectangle that has to land on the grid, and the figure
    is derived from it — a margin of MARGINS[mode] modules on every side,
    which is where the title and the tick labels live.

    So the saved file is (w + 2m) x (h + 2m) modules. Align the inner
    rectangle to a grid cell in InDesign and the margins fall into the
    poster's own gutters.
    """
    m = MARGINS[mode]
    unit = BASELINE_MM / MM_PER_IN                   # one module, in inches
    total_w, total_h = w_modules + 2 * m, h_modules + 2 * m
    fig = plt.figure(figsize=(total_w * unit, total_h * unit))
    ax = fig.add_axes([m / total_w, m / total_h,
                       w_modules / total_w, h_modules / total_h])
    return fig, ax



# A label printed plainly inside this window stays short. Outside it, the
# axis is worth factoring.
PLAIN_LO, PLAIN_HI = 1e-3, 1e5


def factor_exponent(values):
    """The power of ten to divide an axis by so its labels read as numbers.

    A decade ladder written out in full — 1x10^-6, 1x10^-5, 1x10^-4, 1x10^-3
    — is four near-identical strings stacked down the gutter and reads as a
    block of text. Divided by a common factor the same axis becomes 0.01,
    0.1, 1, 10, and the factor is stated once in the corner of the panel.

    Anchored so the LARGEST label comes out under 100, which is what keeps
    the labels short. Centring on the geometric middle instead would give
    3.7, 0.81, 0.19, 0.046 where anchoring gives 37, 8.2, 1.9, 0.46.

    Two exceptions return 0, meaning leave the axis alone: values that are
    already short, and values spanning so many decades that no single factor
    can shorten both ends — there, anchoring the top would drive the bottom
    below 0.01, so the caller falls back to labelling each tick in full.
    """
    if all(PLAIN_LO <= v < PLAIN_HI for v in values):
        return 0
    logs = [math.log10(v) for v in values]
    exponent = math.floor(max(logs)) - 1
    if min(logs) - exponent < -2:
        exponent = math.floor(sum(logs) / len(logs) + 0.5)
    return exponent


def factor_text(exponent):
    """The x10^n label for an axis divided by that power of ten.

    Empty at exponent 0: a factor of 1 multiplies nothing, and printing
    "x10^0" in the corner of a plot is worse than printing nothing.
    """
    return "" if exponent == 0 else rf"$\times10^{{{exponent}}}$"


def factor_label(ax, text=None):
    """Put the y axis's x10^n factor in the top-left gutter, outside the plot.

    `text=None` takes the string from the axis's own ScalarFormatter offset
    and hides that artist. A log axis picks its own factor, so it passes one.
    Returns the annotation, so a live figure can re-set_text it per frame.

    OUTSIDE, not inset. Inside the panel the label sat on the data — measured
    on a real run, 37 samples of one curve passed behind it — and the opaque
    background that kept it legible did so by erasing the line underneath.

    TOP LEFT, one gutter left of the spine and sitting on top of it. The
    factor belongs to the y axis, and that is the corner the y axis starts
    from; in a bottom corner it reads as if the x values were the ones being
    multiplied. At 13 pt "x10^-4" is 29.5 pt against a 26.4 pt gutter, so
    3.1 pt of it overhangs the plot's width — harmless, because it sits above
    the top spine where there is no data.

    It shares that corner with the topmost y tick label, which is why
    rotate_y_labels keeps those inside the panel.
    """
    if text is None:
        ax.figure.canvas.draw()      # the formatter knows its offset only after a draw
        text = ax.yaxis.get_offset_text().get_text()
        ax.yaxis.get_offset_text().set_visible(False)
    return ax.annotate(text, xy=(0, 1), xycoords="axes fraction",
                       xytext=(-MODULE_PT, TICK_PAD_PT),
                       textcoords="offset points", ha="left", va="bottom",
                       color=MUTED, fontsize=TICK_LABEL_PT,
                       annotation_clip=False)


def rotate_y_labels(ax):
    """Turn the y tick labels on their side and keep them inside the panel.

    Rotated, and not for looks: upright, "1x10^-3" is 37.4 pt wide against a
    26.4 pt gutter; on its side it is 13.9 pt and fits.

    rotation_mode="anchor" is required. Without it matplotlib rotates the
    label around its bounding box and the default ha="right" then puts the
    label's top at the tick, so every label hangs below the gridline it
    belongs to. It also aligns BEFORE rotating, which is why va is "bottom":
    once rotated, the text's bottom edge is the one facing the plot, so
    anchoring there is what keeps the whole label in the gutter.

    Then the clamp. A tick sitting exactly on a spine — which is what the
    recorder's log axes do deliberately, snapping their limits outward onto
    the tick set — leaves a centred label hanging half outside the panel, by
    as much as 14.7 pt. Up there it collides with the factor label; down
    below it collides with the x tick labels. Re-aligning drives the label
    off its anchor to one side, and the anchor is the tick itself, so one
    pass is enough to bring the whole label back inside.
    """
    labels = [t for t in ax.get_yticklabels() if t.get_text()]
    plt.setp(labels, rotation=90, ha="center", va="bottom",
             rotation_mode="anchor")

    renderer = ax.figure.canvas.get_renderer()
    panel = ax.get_window_extent()
    for label in labels:
        box = label.get_window_extent(renderer)
        if box.y1 > panel.y1:
            label.set_ha("right")        # drive it down, off the top spine
        elif box.y0 < panel.y0:
            label.set_ha("left")         # drive it up, off the bottom spine


def apply_figure_style(fig, axes, image_axes=()):
    """Apply the ground, rules and type colours to a figure."""
    fig.patch.set_facecolor(GROUND)
    for ax in axes:
        ax.set_facecolor(PANEL if ax not in image_axes else GROUND)
        for s in ax.spines.values():
            s.set_color(RULE)
            s.set_linewidth(SPINE_PT)
        ax.tick_params(colors=MUTED, labelcolor=MUTED)
        ax.title.set_color(TEXT)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.grid(color=RULE, alpha=GRID_ALPHA, linewidth=GRID_PT)


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


def save_exact(fig, path, dpi=300):
    """Save at exactly the figure's declared size — no tight cropping.

    Separate from save() rather than a flag on it, because the two want
    opposite things. save() crops to the ink, which is right for a figure
    that will be looked at on its own and is what replot.py relies on to
    trim a partial layout. A poster figure must come out at the size it was
    built at, or it no longer fits the grid: tight cropping would make the
    page size depend on how long the tick labels happened to be.
    """
    fig.savefig(path, dpi=dpi, facecolor=GROUND, edgecolor="none")