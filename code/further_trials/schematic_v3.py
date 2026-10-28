"""Design schematic for Figure 1a, three layouts with simple drawn icons (no icon fonts or external images).
Drawn in inch units on an axes with equal aspect, so shapes keep their proportions.

  A  icons in a row with the labels underneath (classic schematic)
  B  icon beside its label, compact single row
  C  icons and labels in light cards, compact

Content, as agreed: manuscripts (3 test papers from 2004, 58 trials published in 2026, 14 inserted errors) ->
15 LLMs (no tools or web access) -> 2 LLM judges (fixed answer key) -> major and minor errors (sensitivity and
specificity).
"""
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

INK, INK2, LINE, FILL = "#1A1A1A", "#555555", "#8A8A8A", "#EEF0F2"
SLATE, ROSE, HL = "#4F6375", "#C47885", "#E9C46A"     # proprietary, open-weight, highlighted (inserted) errors
FILLS_C = ["#F6F7F8", "#EEF0F3", "#E6E9ED", "#DEE2E7"]  # option C cards, light to slightly darker
N_OPEN = 4

HEADS = ["Manuscripts", "15 LLMs", "2 LLM judges", "Major and minor errors"]
SUBS = [["3 test papers (2004)", "58 trials (2026)", "14 inserted errors"],
        ["no tools or web access"],
        ["fixed answer key"],
        ["sensitivity and specificity"]]


# ------------------------------------------------------------------------------------------------ icons
def _page(ax, x, y, w, h, hl=(), n=7, z=2):
    """Manuscript page (lower-left corner x, y) with a folded corner, text lines and highlighted lines."""
    f = 0.22 * w
    ax.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + h - f), (x + w - f, y + h), (x, y + h)], closed=True,
                         fc="white", ec=LINE, lw=0.6, zorder=z, joinstyle="round"))
    ax.add_patch(Polygon([(x + w - f, y + h), (x + w - f, y + h - f), (x + w, y + h - f)], closed=True,
                         fc="#E6E6E6", ec=LINE, lw=0.5, zorder=z + 0.1))
    top, bot = y + h - 0.20 * h, y + 0.13 * h
    for k, ly in enumerate(np.linspace(top, bot, n)):
        x1 = x + w * (0.82 if k == 0 else 0.86 if k % 3 else 0.62)
        if k in hl:
            ax.add_patch(Rectangle((x + 0.12 * w, ly - 0.022), x1 - x - 0.12 * w, 0.044, fc=HL, ec="none",
                                   zorder=z + 0.2))
        ax.plot([x + 0.14 * w, x1 - 0.02 * w], [ly, ly], color="#B5B5B5", lw=0.7, solid_capstyle="round",
                zorder=z + 0.3)


def icon_manuscripts(ax, cx, cy, s):
    w, h = 0.62 * s, 0.82 * s
    _page(ax, cx - w / 2 - 0.10 * s, cy - h / 2 + 0.07 * s, w, h, z=2)
    _page(ax, cx - w / 2 + 0.10 * s, cy - h / 2 - 0.07 * s, w, h, hl=(1, 3, 5), z=3)


def icon_llms(ax, cx, cy, s):
    cols, rows = 5, 3
    gap = 0.19 * s
    r = 0.068 * s
    k = 0
    for j in range(rows):
        for i in range(cols):
            x = cx + (i - (cols - 1) / 2) * gap
            y = cy + ((rows - 1) / 2 - j) * gap
            ax.add_patch(Circle((x, y), r, fc=ROSE if k >= cols * rows - N_OPEN else SLATE, ec="white", lw=0.4,
                                zorder=3))
            k += 1


def _check(ax, x, y, s, ok=True):
    if ok:
        ax.plot([x - 0.05 * s, x - 0.012 * s, x + 0.06 * s], [y, y - 0.04 * s, y + 0.05 * s], color=INK2, lw=0.9,
                solid_capstyle="round", solid_joinstyle="round", zorder=5)
    else:
        d = 0.04 * s
        ax.plot([x - d, x + d], [y - d, y + d], color=INK2, lw=0.9, solid_capstyle="round", zorder=5)
        ax.plot([x - d, x + d], [y + d, y - d], color=INK2, lw=0.9, solid_capstyle="round", zorder=5)


def icon_judges(ax, cx, cy, s):
    w, h = 0.46 * s, 0.70 * s
    for k, dx in enumerate((-0.27 * s, 0.27 * s)):
        x0, y0 = cx + dx - w / 2, cy - h / 2
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={0.05 * s}", fc="white",
                                    ec=LINE, lw=0.6, zorder=3))
        ax.add_patch(FancyBboxPatch((cx + dx - 0.10 * s, y0 + h - 0.04 * s), 0.20 * s, 0.08 * s,
                                    boxstyle=f"round,pad=0,rounding_size={0.02 * s}", fc=LINE, ec="none", zorder=4))
        marks = (True, True, False) if k == 0 else (True, False, True)
        for j, ok in enumerate(marks):
            ly = y0 + h - 0.20 * s - j * 0.17 * s
            _check(ax, x0 + 0.12 * s, ly, s * 0.9, ok)
            ax.plot([x0 + 0.21 * s, x0 + w - 0.06 * s], [ly, ly], color="#B5B5B5", lw=0.7, solid_capstyle="round",
                    zorder=4)


def icon_metrics(ax, cx, cy, s):
    """Mini sensitivity-specificity plot, as in Fig. 2d: axes, the human reference line and models top right."""
    w, h = 0.78 * s, 0.70 * s
    x0, y0 = cx - w / 2, cy - h / 2
    ax.plot([x0, x0, x0 + w], [y0 + h, y0, y0], color=INK2, lw=0.7, solid_capstyle="round", zorder=3)
    ax.plot([x0 + 0.27 * w] * 2, [y0 + 0.04 * h, y0 + h], color=LINE, lw=0.6, ls=(0, (2, 1.5)), zorder=3)
    rng = [(0.55, 0.78), (0.68, 0.84), (0.78, 0.72), (0.88, 0.66), (0.62, 0.70), (0.93, 0.60)]
    for k, (u, v) in enumerate(rng):
        ax.add_patch(Circle((x0 + u * w, y0 + v * h), 0.045 * s, fc=ROSE if k in (1, 4) else SLATE, ec="white",
                            lw=0.3, zorder=4))


ICONS = [icon_manuscripts, icon_llms, icon_judges, icon_metrics]


def _arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=7, lw=0.7, color=LINE,
                                 shrinkA=0, shrinkB=0, zorder=2))


def _text_w(ax, txt, fs, bold=False):
    fig = ax.figure
    t = ax.text(0, 0, txt, fontsize=fs, fontweight="bold" if bold else "normal")
    fig.canvas.draw()
    bb = t.get_window_extent(fig.canvas.get_renderer())
    t.remove()
    return bb.width / fig.dpi


# ---------------------------------------------------------------------------------------------- layouts
def draw_A(ax, W, H, fs_head=7.5, fs_sub=7):
    """Icons in a row, labels underneath."""
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.set_axis_off()
    s = 0.50
    icon_y = H - 0.05 - s * 0.45
    xs = [W * (k + 0.5) / 4 for k in range(4)]
    for k in range(4):
        ICONS[k](ax, xs[k], icon_y, s)
        ax.text(xs[k], icon_y - s * 0.55 - 0.05, HEADS[k], ha="center", va="top", fontsize=fs_head,
                fontweight="bold", color=INK)
        for j, sub in enumerate(SUBS[k]):
            ax.text(xs[k], icon_y - s * 0.55 - 0.20 - j * 0.125, sub, ha="center", va="top", fontsize=fs_sub,
                    color=INK2)
    for k in range(3):
        _arrow(ax, xs[k] + 0.42, xs[k + 1] - 0.42, icon_y)


def draw_B(ax, W, H, fs_head=7.5, fs_sub=7):
    """Icon beside its label, all four steps in one compact row."""
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.set_axis_off()
    s = 0.46
    iw, pad, arr = 0.52, 0.07, 0.30
    tw = [max([_text_w(ax, HEADS[k], fs_head, True)] + [_text_w(ax, t, fs_sub) for t in SUBS[k]]) for k in range(4)]
    blocks = [iw + pad + t for t in tw]
    gap = (W - sum(blocks)) / 3
    cy = H / 2
    x = 0
    for k in range(4):
        ICONS[k](ax, x + iw / 2, cy, s)
        lines = [(HEADS[k], True)] + [(t, False) for t in SUBS[k]]
        step = 0.125
        y0 = cy + step * (len(lines) - 1) / 2
        for j, (txt, b) in enumerate(lines):
            ax.text(x + iw + pad, y0 - j * step, txt, ha="left", va="center", fontsize=fs_head if b else fs_sub,
                    fontweight="bold" if b else "normal", color=INK if b else INK2)
        x_end = x + blocks[k]
        if k < 3:
            mid = x_end + gap / 2
            _arrow(ax, mid - min(arr, gap * 0.7) / 2, mid + min(arr, gap * 0.7) / 2, cy)
        x = x_end + gap


def draw_C(ax, W, H, fs_head=7.5, fs_sub=7):
    """Icon and label in light cards sized to their content; thin arrows between the cards."""
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.set_axis_off()
    s = 0.42
    iw, pad, gap = 0.50, 0.06, 0.24
    tw = [max([_text_w(ax, HEADS[k], fs_head, True)] + [_text_w(ax, t, fs_sub) for t in SUBS[k]]) for k in range(4)]
    inner = [iw + pad + t for t in tw]
    extra = (W - 3 * gap - sum(inner)) / 4          # spread evenly as margins inside each card
    x0 = 0
    for k in range(4):
        w = inner[k] + extra
        ax.add_patch(FancyBboxPatch((x0, 0.0), w, H, boxstyle="round,pad=0,rounding_size=0.05", fc=FILLS_C[k],
                                    ec="none", zorder=1))
        ix = x0 + extra / 2 + iw / 2
        ICONS[k](ax, ix, H / 2, s)
        lines = [(HEADS[k], True)] + [(t, False) for t in SUBS[k]]
        step = 0.125
        y0 = H / 2 + step * (len(lines) - 1) / 2
        for j, (txt, b) in enumerate(lines):
            ax.text(ix + iw / 2 + pad, y0 - j * step, txt, ha="left", va="center", fontsize=fs_head if b else fs_sub,
                    fontweight="bold" if b else "normal", color=INK if b else INK2, zorder=3)
        if k < 3:
            _arrow(ax, x0 + w + 0.04, x0 + w + gap - 0.04, H / 2)
        x0 += w + gap


DRAW = {"A": (draw_A, 1.12), "B": (draw_B, 0.62), "C": (draw_C, 0.62)}    # function, panel height (in)
