"""Drawing helpers shared by the reports. Colours are read from style at call time, so every report follows the style."""
import re

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import Rectangle
from mplsoccer import Pitch, VerticalPitch

from .. import style


def home_col():
    return style.TEAL


def away_col():
    return style.BLUE


def side_col(is_home):
    return home_col() if is_home else away_col()


def pitch_kw(lw=1.0):
    return dict(pitch_type="custom", pitch_length=105, pitch_width=68, pitch_color=style.BG, line_color=style.HAIR,
                linewidth=lw, line_zorder=2)


def pitch(ax, half=False, vertical=False):
    if half:
        pt = VerticalPitch(half=True, pad_bottom=0.5, **pitch_kw())
    else:
        pt = (VerticalPitch if vertical else Pitch)(**pitch_kw())
    pt.draw(ax=ax)
    return pt


def px(x):
    return np.asarray(x, dtype=float) + 52.5


def py(y):
    return np.asarray(y, dtype=float) + 34


def cmap_to(color):
    return LinearSegmentedColormap.from_list("t", [style.BG, color])


def lava():
    return LinearSegmentedColormap.from_list("lava", [style.BG, style.DIM, style.GOLD, style.ACCENT, style.DANGER_TOP])


def matchday(md):
    if not md:
        return None
    m = re.match(r"(\d+)", str(md))
    return f"Matchday {m.group(1)}" if m else str(md)


def clean(ax, grid="x"):
    style.clean(ax, grid=grid)


def axis_off(ax):
    ax.set_axis_off()
    ax.set_facecolor(style.BG)


def text(ax, x, y, s, size=9, color=None, bold=False, serif=False, ha="left", va="center", **kw):
    return ax.text(x, y, s, fontsize=size, color=color or style.INK, weight="bold" if bold else "normal",
                   family=style.SERIF if serif else style.SANS, ha=ha, va=va, **kw)


def section(ax, x, y, label, color=None):
    ax.scatter([x], [y], s=26, marker="s", color=color or style.ACCENT, transform=ax.transAxes, clip_on=False)
    ax.text(x + 0.015, y, label.upper(), transform=ax.transAxes, fontsize=8, weight="bold", color=style.INK, va="center")


def compare_bars(ax, rows, home_name, away_name):
    """rows: (label, home value, away value, fmt). Two bars per metric, sized by each side's share."""
    axis_off(ax)
    n = len(rows)
    ax.set_xlim(0, 1), ax.set_ylim(n, -0.6)
    hc, ac = home_col(), away_col()
    text(ax, 0.0, -0.45, home_name, size=8.5, bold=True, color=hc)
    text(ax, 1.0, -0.45, away_name, size=8.5, bold=True, color=ac, ha="right")
    for i, (lab, hv, av, fmt) in enumerate(rows):
        y = i + 0.35
        tot = (abs(hv) if np.isfinite(hv) else 0) + (abs(av) if np.isfinite(av) else 0)
        hs = abs(hv) / tot if tot else 0.5
        text(ax, 0.5, y - 0.18, lab, size=7.8, color=style.MUTED2, ha="center")
        text(ax, 0.0, y - 0.18, "-" if not np.isfinite(hv) else fmt.format(hv), size=8.6, bold=True)
        text(ax, 1.0, y - 0.18, "-" if not np.isfinite(av) else fmt.format(av), size=8.6, bold=True, ha="right")
        ax.barh(y + 0.17, hs, left=0, height=0.16, color=hc)
        ax.barh(y + 0.17, 1 - hs - 0.004, left=hs + 0.004, height=0.16, color=ac)


def hbars(ax, labels, values, colors, fmt="{:.2f}", vmax=None, label_size=8):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colors, height=0.62)
    lim = vmax or (max(np.nanmax(np.abs(values)), 1e-9) * 1.25 if len(values) else 1)
    for yi, v in zip(y, values):
        text(ax, v + lim * 0.015 if v >= 0 else v - lim * 0.015, yi, fmt.format(v), size=7.6,
             ha="left" if v >= 0 else "right")
    ax.set_yticks(y), ax.set_yticklabels(labels, fontsize=label_size, color=style.INK)
    lo = min(0, np.nanmin(values) * 1.25) if len(values) else 0
    ax.set_xlim(lo, lim)
    clean(ax)


def table(ax, header, rows, widths, fmts, colours=None, size=8.0, head_size=7.2):
    """Plain table on an axis-off panel: text columns (fmt "{}") left-aligned, numbers right-aligned.
    colours: optional per-cell (color, alpha) or None."""
    axis_off(ax)
    n = len(rows)
    ax.set_xlim(0, 1), ax.set_ylim(n + 0.2, -1.0)
    xs = np.cumsum([0] + list(widths[:-1]))
    left = [f == "{}" for f in fmts]
    for x, h, w, lft in zip(xs, header, widths, left):
        text(ax, x + 0.004 if lft else x + w - 0.006, -0.5, h.upper(), size=head_size, bold=True, color=style.MUTED2,
             ha="left" if lft else "right")
    ax.plot([0, 1], [-0.12, -0.12], color=style.HAIR, lw=0.8)
    for i, r in enumerate(rows):
        for j, (x, w, v, f, lft) in enumerate(zip(xs, widths, r, fmts, left)):
            c = colours[i][j] if colours else None
            if c:
                ax.add_patch(Rectangle((x + 0.002, i + 0.06), w - 0.004, 0.88, color=c[0], alpha=c[1], lw=0))
            s = f.format(v) if (v is not None and not (isinstance(v, float) and np.isnan(v))) else "-"
            text(ax, x + 0.004 if lft else x + w - 0.006, i + 0.5, s, size=size, ha="left" if lft else "right",
                 color=style.ink_on(*c) if c else style.INK)


def dot_legend(fig_ax, x, y, items, size=8):
    """items: (label, color, marker). Drawn in axes fraction coordinates."""
    for lab, col, mk in items:
        fig_ax.scatter([x], [y], s=40, color=col, marker=mk, transform=fig_ax.transAxes, clip_on=False,
                       edgecolor=style.BG if mk != "o" else None)
        fig_ax.text(x + 0.012, y, lab, transform=fig_ax.transAxes, fontsize=size, color=style.MUTED2, va="center")
        x += 0.03 + 0.0062 * len(lab)


def alpha(c, a):
    return to_rgba(c, a)


def plural(n, word, many=None):
    n = int(n)
    return f"{n} {word if n == 1 else (many or word + 's')}"
