"""Waltzing Analytics page frame for single-chart exports (layout in inches), in seven colour themes.

The six house themes form a temperature x lightness grid (cool: fog, dusk, navy; warm: cream, clay, espresso) with the
coral accent held constant; xhotspur is the Tottenham club theme (Spurs navy as the accent, coral kept for the mark).
use("fog") switches every chart module at once; navy is the default.
"""
import sys
from datetime import date
from pathlib import Path

from matplotlib import font_manager

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb
from matplotlib.patches import FancyBboxPatch, Rectangle

from . import config


def _mix(a, b, t):
    return to_hex([x + (y - x) * t for x, y in zip(to_rgb(a), to_rgb(b))])


# canvas, ink, muted (footer), muted2 (labels), hairline, accent, below-average/secondary series (blue),
# third series (teal), fourth series (gold); anything missing is derived in _tokens()
THEMES = {
    "navy": dict(BG="#14202B", INK="#F2EFE6", MUTED="#86847E", MUTED2="#A8A6A0", HAIR="#384653", ACCENT="#E8794C",
                 BLUE="#5B8AC9", TEAL="#1FA88C", GOLD="#B37F1E", CARD="#1C2935", GREY="#3C4A57", SLATE="#6B7884",
                 DIM="#2A3A48", DIM2="#2E3B48", DANGER_TOP="#F7D8C6"),
    "fog": dict(BG="#E7ECEE", INK="#1A2226", MUTED="#78888D", HAIR="#CBD6D9", ACCENT="#E8794C",
                BLUE="#4C7A93", TEAL="#5E8C6A", GOLD="#C0902F"),
    "dusk": dict(BG="#4A5A66", INK="#EDF1F2", MUTED="#A9BAC0", MUTED2="#CBD8DB", HAIR="#6B7D85", ACCENT="#E8794C",
                 BLUE="#7B7FB0", TEAL="#6FA68C", GOLD="#D9A84E"),
    "cream": dict(BG="#F5F1E8", INK="#20201C", MUTED="#8B8880", HAIR="#DFD9C8", ACCENT="#E8794C",
                  BLUE="#8A8F5C", TEAL="#4C8573", GOLD="#C08A2E"),
    "clay": dict(BG="#5C4E41", INK="#F1ECDF", MUTED="#B4A794", MUTED2="#D6CAB5", HAIR="#7C6C57", ACCENT="#E8794C",
                 BLUE="#8FA6B8", TEAL="#6B8F6A", GOLD="#D2A24C"),
    "espresso": dict(BG="#1E1811", INK="#F1ECDF", MUTED="#9A8C77", MUTED2="#C4B9A5", HAIR="#3C3122", ACCENT="#E8794C",
                     BLUE="#4A8C82", TEAL="#7FA3B8", GOLD="#C99A3E"),
    "xhotspur": dict(BG="#F7F8FA", INK="#132257", MUTED="#6B7591", MUTED2="#4F5A78", HAIR="#D5DAE6", ACCENT="#132257",
                     BLUE="#8FA0CC", TEAL="#4F6DB8", GOLD="#C9A227", GREY="#C9D0E0", BAND="#132257"),
}
LIGHT = {"fog", "cream", "xhotspur"}


def _tokens(name):
    t = dict(THEMES[name])
    bg, ink, light = t["BG"], t["INK"], name in LIGHT
    t.setdefault("MUTED2", _mix(ink, bg, 0.38))       # light canvases: labels need more contrast than the footer grey
    t.setdefault("CARD", _mix(bg, "#FFFFFF", 0.55) if light else _mix(bg, ink, 0.04))
    t.setdefault("GREY", _mix(bg, ink, 0.16))         # context bars and points
    t.setdefault("SLATE", _mix(bg, ink, 0.40))
    t.setdefault("DIM", _mix(bg, ink, 0.09))
    t.setdefault("DIM2", t["DIM"])
    t.setdefault("DANGER_TOP", _mix(t["ACCENT"], ink, 0.35 if light else 0.55))
    t.setdefault("RED", "#D9663A")
    t.setdefault("PURPLE", "#8C7BD1")
    t.setdefault("MARK_COLOR", "#E8794C")             # the mark stays coral in every theme
    t.setdefault("BAND", None)
    t["DANGER_CMAP"] = LinearSegmentedColormap.from_list("wa_danger", [bg, t["GREY"], t["GOLD"], t["ACCENT"], t["DANGER_TOP"]])
    t["COMPONENTS"] = [("pass", "Passes", t["BLUE"]), ("carry", "Carries", t["TEAL"]), ("other", "Restarts, other", t["SLATE"])]
    return t


def _lum(c):
    lin = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in to_rgb(c)]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def ink_on(color, alpha=1.0):
    """Text colour for a value printed on a filled cell: the theme ink, unless it would vanish into the cell
    (e.g. navy text on a navy cell in xhotspur); then whichever of ink and canvas reads better."""
    cell = _mix(BG, color, alpha)
    ratio = lambda a, b: (max(_lum(a), _lum(b)) + 0.05) / (min(_lum(a), _lum(b)) + 0.05)
    if ratio(INK, cell) >= 2.2:
        return INK
    return INK if ratio(INK, cell) >= ratio(BG, cell) else BG


def text_on(color):
    """Canvas or ink, whichever reads better on a solid fill (pizza and bar labels)."""
    ratio = lambda a, b: (max(_lum(a), _lum(b)) + 0.05) / (min(_lum(a), _lum(b)) + 0.05)
    return BG if ratio(BG, color) >= ratio(INK, color) else INK


_COLLECT = []


class collect:
    """with collect() as pages: chart functions that call Page.save hand their pages to the list instead."""
    def __enter__(self):
        _COLLECT.append([])
        return _COLLECT[-1]

    def __exit__(self, *exc):
        _COLLECT.pop()
        return False


THEME = config.STYLE.lower() if config.STYLE.lower() in THEMES else "navy"
globals().update(_tokens(THEME))
_HOOKS = []


def on_theme(fn):
    """Register a module function that rebuilds colour constants derived from the tokens (pitch dicts, colour maps)."""
    _HOOKS.append(fn)
    fn()
    return fn


def themes():
    return list(THEMES)


def use(name):
    """Switch every chart module to theme name (navy, fog, dusk, cream, clay, espresso, xhotspur)."""
    global THEME
    name = name.lower()
    if name not in THEMES:
        raise ValueError(f"Unknown style '{name}'. Styles: {', '.join(THEMES)}")
    THEME = name
    t = _tokens(name)
    globals().update(t)
    for mod in list(sys.modules.values()):
        if getattr(mod, "__name__", "").startswith("wa_restarts.") and mod not in (sys.modules[__name__], config):
            for k, v in t.items():
                if k in vars(mod):
                    setattr(mod, k, v)
    _rc()
    for fn in _HOOKS:
        fn()


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
# font pairings (headline serif, body sans). house uses the macOS fonts; the others ship with the package
# (SIL Open Font License, wa_restarts/fonts_bundled/) so they look the same on every machine.
for _f in (Path(__file__).parent / "fonts_bundled").glob("*.ttf"):
    font_manager.fontManager.addfont(str(_f))
FONTS = {
    "house": (("Georgia", "Times New Roman", "Liberation Serif", "DejaVu Serif"),
              ("Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans")),
    "plex": (("IBM Plex Serif",), ("Lato",)),
    "spectral": (("Spectral",), ("Barlow",)),
    "display": (("DM Serif Display",), ("Lato",)),
    "barlow": (("Barlow",), ("Barlow",)),
}
FONT = config.FONT.lower() if config.FONT.lower() in FONTS else "house"
SERIF, SANS = _first(*FONTS[FONT][0]), _first(*FONTS[FONT][1])


def fonts():
    return list(FONTS)


def use_font(name):
    """Switch the headline / body font pairing (house, plex, spectral, display, barlow)."""
    global FONT, SERIF, SANS
    name = name.lower()
    if name not in FONTS:
        raise ValueError(f"Unknown font '{name}'. Fonts: {', '.join(FONTS)}")
    FONT, SERIF, SANS = name, _first(*FONTS[name][0]), _first(*FONTS[name][1])
    for mod in list(sys.modules.values()):
        if getattr(mod, "__name__", "").startswith("wa_restarts.") and mod not in (sys.modules[__name__], config):
            for k, v in (("SERIF", SERIF), ("SANS", SANS)):
                if k in vars(mod):
                    setattr(mod, k, v)
    _rc()



def _rc():
    plt.rcParams.update({"font.family": SANS, "text.color": INK, "axes.labelcolor": MUTED2,
                         "xtick.color": MUTED2, "ytick.color": MUTED2, "axes.edgecolor": HAIR,
                         "axes.facecolor": BG, "figure.facecolor": BG, "axes.labelsize": 9.5,
                         "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "figure.max_open_warning": 0})  # reports hold every page open until the PDF is written


_rc()


class Page:
    def __init__(self, h, eyebrow, title, subtitle, source=None, w=None):
        source = source or config.source()
        self.h, self.w = h, w or globals()["W"]
        W = self.w
        self.fig = plt.figure(figsize=(W, h), facecolor=BG)
        bg = self.fig.add_axes([0, 0, 1, 1])
        bg.set_axis_off()
        bg.set_xlim(0, 1), bg.set_ylim(0, 1)
        bg.set_autoscale_on(False)
        self.bg = bg
        y = lambda inch: 1 - inch / h
        if BAND:  # club themes: a thin band in the club colour across the top edge
            bg.add_patch(Rectangle((0, y(0.07)), 1, 0.07 / h, color=BAND, lw=0, transform=bg.transAxes))
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
        bg.text(xr - 1.74 / W, y(0.6), config.MARK, color=MARK_COLOR, fontsize=24, weight="bold", ha="right", va="center")
        bg.plot([xr - 1.52 / W] * 2, [y(0.4), y(0.76)], color=HAIR, lw=1, transform=bg.transAxes)
        bg.text(xr, y(0.52), config.BRAND.upper(), fontsize=9, weight="bold", ha="right", va="center", color=INK)
        bg.text(xr, y(0.72), config.AUTHOR.upper(), fontsize=7, ha="right", va="center", color=MUTED)
        # footer
        bg.plot([L / W, xr], [0.62 / h] * 2, color=HAIR, lw=0.8, transform=bg.transAxes)
        narrow = W < 9  # portrait pages: source and author on two lines
        bg.text(L / W, (0.45 if narrow else 0.34) / h, source, fontsize=7.5 if narrow else 8, color=MUTED, va="center")
        bg.text(xr, (0.22 if narrow else 0.34) / h, f"{config.AUTHOR} | Created on {date.today():%d-%m-%Y}", fontsize=8,
                color=MUTED, ha="right", va="center")

    def axes(self, top, bottom, left=L, right=R, **kw):
        W = self.w
        """Axes placed by inches from the top edge (top) and the bottom edge (bottom)."""
        ax = self.fig.add_axes([left / W, bottom / self.h, 1 - (left + right) / W,
                                (self.h - top - bottom) / self.h], **kw)
        ax.set_facecolor(BG)
        return ax

    def section(self, label, top, left=L):
        W = self.w
        y = 1 - top / self.h
        self.bg.add_patch(Rectangle((left / W, y - 0.04 / self.h), 0.08 / W, 0.08 / self.h,
                                    color=ACCENT, transform=self.bg.transAxes))
        self.bg.text((left + 0.15) / W, y, label.upper(), fontsize=8.5, weight="bold", color=INK, va="center")

    def cards(self, items, top, height=0.95, left=L, right=R, gap=0.14):
        W = self.w
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
        W = self.w
        x = left
        y = 1 - top / self.h
        for label, color in entries:
            self.bg.scatter([(x + 0.06) / W], [y], s=55, marker=marker, color=color, transform=self.bg.transAxes)
            self.bg.text((x + 0.18) / W, y, label, fontsize=8.5, color=MUTED2, va="center")
            x += 0.4 + 0.068 * len(label)

    def save(self, path):
        if _COLLECT:  # inside collect(): hand the page to a report instead of writing a PNG
            _COLLECT[-1].append((self, Path(path).stem))
            return
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
