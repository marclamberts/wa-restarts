"""Waltzing Analytics dark navy page frame for single-chart exports (layout in inches)."""
from datetime import date
from pathlib import Path

from matplotlib import font_manager

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle

from . import config

BG, CARD, INK, MUTED, MUTED2, HAIR = "#14202B", "#1C2935", "#F2EFE6", "#86847E", "#A8A6A0", "#384653"
ACCENT, TEAL, BLUE, GOLD, RED, GREY = "#E8794C", "#1FA88C", "#5B8AC9", "#B37F1E", "#D9663A", "#3C4A57"
# xDanger added sources: passes include the reception they set up; shots sit outside (shot selection)
COMPONENTS = [("pass", "Passes", BLUE), ("carry", "Carries", TEAL), ("other", "Restarts, other", "#6B7884")]
DANGER_CMAP = LinearSegmentedColormap.from_list("wa_danger", [BG, GREY, GOLD, ACCENT, "#F7D8C6"])
W = 11.0
L, R = 0.55, 0.55

# Helvetica Neue and Georgia ship with macOS; matplotlib only reads the first face of the system .ttc, so put extracted
# .ttf faces in wa_restarts/fonts/ (or point WA_RESTARTS_FONTS at a folder). Elsewhere the nearest installed font is used.
import os  # noqa: E402

for _d in (Path(__file__).parent / "fonts", Path(os.getenv("WA_RESTARTS_FONTS", Path(__file__).parent / "fonts"))):
    for _f in _d.glob("*.ttf"):
        font_manager.fontManager.addfont(str(_f))


def _first(*names):
    have = {f.name for f in font_manager.fontManager.ttflist}
    return next((n for n in names if n in have), names[-1])


def _nan_safe_bins():
    """scipy 1.18 rejects NaN coordinates in binned statistics; older versions left those points out of every bin.
    Drop them before binning so heat maps behave the same on every scipy version."""
    import numpy as np
    from mplsoccer import _pitch_base

    orig = _pitch_base.bin_statistic
    if getattr(orig, "_nan_safe", False):
        return

    def bin_statistic(x, y, values=None, **kw):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        ok = ~(np.isnan(x) | np.isnan(y))
        if not ok.all():
            x, y = x[ok], y[ok]
            values = None if values is None else np.asarray(values, dtype=float)[ok]
        return orig(x, y, values=values, **kw)

    bin_statistic._nan_safe = True
    _pitch_base.bin_statistic = bin_statistic


_nan_safe_bins()
SANS = _first("Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans")
SERIF = _first("Georgia", "Times New Roman", "Liberation Serif", "DejaVu Serif")

plt.rcParams.update({"font.family": SANS, "text.color": INK, "axes.labelcolor": MUTED2,
                     "xtick.color": MUTED2, "ytick.color": MUTED2, "axes.edgecolor": HAIR,
                     "axes.facecolor": BG, "figure.facecolor": BG, "axes.labelsize": 9.5,
                     "xtick.labelsize": 9, "ytick.labelsize": 9})


class Page:
    def __init__(self, h, eyebrow, title, subtitle, source="Data: IMPECT, all leagues 2024 to 2027  ·  xG and xDanger: Waltzing Analytics"):
        self.h = h
        self.fig = plt.figure(figsize=(W, h), facecolor=BG)
        bg = self.fig.add_axes([0, 0, 1, 1])
        bg.set_axis_off()
        bg.set_xlim(0, 1), bg.set_ylim(0, 1)
        bg.set_autoscale_on(False)
        self.bg = bg
        y = lambda inch: 1 - inch / h
        bg.add_patch(Rectangle((L / W, y(0.52) - 0.04 / h), 0.09 / W, 0.09 / h, color=ACCENT, transform=bg.transAxes))
        bg.text((L + 0.17) / W, y(0.48), eyebrow.upper(), color=ACCENT, fontsize=9.5, weight="bold", va="center")
        t = bg.text(L / W, y(0.95), title, family=SERIF, weight="bold", fontsize=21, color=INK, va="center")
        # shrink a long headline until it fits inside the right margin
        renderer = self.fig.canvas.get_renderer()
        limit = self.fig.bbox.width * (1 - R / W)
        while t.get_window_extent(renderer).x1 > limit and t.get_fontsize() > 14:
            t.set_fontsize(t.get_fontsize() - 0.5)
        bg.text(L / W, y(1.33), subtitle, fontsize=10.5, color=MUTED2, va="center")
        # WA lockup: mark, divider, wordmark
        xr = 1 - R / W
        bg.text(xr - 1.74 / W, y(0.6), config.MARK, color=ACCENT, fontsize=24, weight="bold", ha="right", va="center")
        bg.plot([xr - 1.52 / W] * 2, [y(0.4), y(0.76)], color=HAIR, lw=1, transform=bg.transAxes)
        bg.text(xr, y(0.52), config.BRAND.upper(), fontsize=9, weight="bold", ha="right", va="center", color=INK)
        bg.text(xr, y(0.72), config.AUTHOR.upper(), fontsize=7, ha="right", va="center", color=MUTED)
        # footer
        bg.plot([L / W, xr], [0.62 / h] * 2, color=HAIR, lw=0.8, transform=bg.transAxes)
        bg.text(L / W, 0.34 / h, source, fontsize=8, color=MUTED, va="center")
        bg.text(xr, 0.34 / h, f"{config.AUTHOR} | Created on {date.today():%d-%m-%Y}", fontsize=8,
                color=MUTED, ha="right", va="center")

    def axes(self, top, bottom, left=L, right=R, **kw):
        """Axes placed by inches from the top edge (top) and the bottom edge (bottom)."""
        ax = self.fig.add_axes([left / W, bottom / self.h, 1 - (left + right) / W,
                                (self.h - top - bottom) / self.h], **kw)
        ax.set_facecolor(BG)
        return ax

    def section(self, label, top, left=L):
        y = 1 - top / self.h
        self.bg.add_patch(Rectangle((left / W, y - 0.04 / self.h), 0.08 / W, 0.08 / self.h,
                                    color=ACCENT, transform=self.bg.transAxes))
        self.bg.text((left + 0.15) / W, y, label.upper(), fontsize=8.5, weight="bold", color=INK, va="center")

    def cards(self, items, top, height=0.95, left=L, right=R, gap=0.14):
        """items: (label, value, sub). Drawn as dark panels with hairline borders."""
        n = len(items)
        w = (W - left - right - gap * (n - 1)) / n
        for i, (label, value, sub) in enumerate(items):
            x0 = left + i * (w + gap)
            yb = self.h - top - height
            self.bg.add_patch(FancyBboxPatch((x0 / W, yb / self.h), w / W, height / self.h,
                                             boxstyle="round,pad=0,rounding_size=0.004", fc=CARD,
                                             ec=HAIR, lw=0.8, transform=self.bg.transAxes))
            tx = (x0 + 0.14) / W
            self.bg.text(tx, (yb + height - 0.2) / self.h, label.upper(), fontsize=7, weight="bold", color=MUTED2, va="center")
            self.bg.text(tx, (yb + height * 0.45) / self.h, value, family=SERIF, fontsize=17, weight="bold", color=INK, va="center")
            self.bg.text(tx, (yb + 0.15) / self.h, sub, fontsize=7, color=MUTED, va="center")

    def legend(self, entries, top, left=L, marker="s"):
        x = left
        y = 1 - top / self.h
        for label, color in entries:
            self.bg.scatter([(x + 0.06) / W], [y], s=55, marker=marker, color=color, transform=self.bg.transAxes)
            self.bg.text((x + 0.18) / W, y, label, fontsize=8.5, color=MUTED2, va="center")
            x += 0.4 + 0.068 * len(label)

    def save(self, path):
        self.fig.savefig(path, dpi=200, facecolor=BG)
        plt.close(self.fig)


def clean(ax, grid="x"):
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(sp == "bottom")
    ax.spines["bottom"].set_color(HAIR)
    ax.tick_params(length=0)
    if grid:
        ax.grid(axis=grid, color=HAIR, lw=0.5, alpha=0.7)
        ax.set_axisbelow(True)
