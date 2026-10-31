"""Pitch page helpers from xDanger 10_pitch_visuals.py."""
from matplotlib import cm
from mplsoccer import Pitch, VerticalPitch

from . import config, style
from .style import BG, HAIR, MUTED2, L, W, Page


@style.on_theme
def _theme():
    global PITCH
    PITCH = dict(pitch_type="custom", pitch_length=105, pitch_width=68, pitch_color=style.BG, line_color=style.HAIR,
                 linewidth=1, line_zorder=3)


def page(title, eyebrow, sub, kind="full", cards=None, legend=None, cbar=None, src=None, ncols=1):
    """Page with one or more pitches. kind: full | half (vertical attacking half)."""
    pitch_h = {"full": 6.35, "half": 5.6}[kind]
    top = 1.75 + (1.15 if cards else 0) + (0.4 if legend else 0)
    bottom = 0.85 + (0.6 if cbar else 0)
    p = Page(top + pitch_h + bottom, eyebrow, title, sub, source=src)
    if cards:
        p.cards(cards, top=1.7, height=0.95)
    if legend:
        p.legend(legend, top=top - 0.22, marker="o")
    pt = Pitch(**PITCH) if kind == "full" else VerticalPitch(half=True, pad_bottom=0.5, **PITCH)
    axes = []
    gap = 0.3
    w = (W - 2 * L - gap * (ncols - 1)) / ncols
    for k in range(ncols):
        left = L + k * (w + gap)
        ax = p.axes(top=top, bottom=bottom, left=left, right=W - left - w)
        pt.draw(ax=ax)
        axes.append(ax)
    return p, pt, (axes if ncols > 1 else axes[0])


def colorbar(p, cmap, norm, label, fmt="{:.3f}"):
    cax = p.axes(top=p.h - 1.25, bottom=1.08, left=3.4, right=3.4)
    cb = p.fig.colorbar(cm.ScalarMappable(norm, cmap), cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cax.tick_params(length=0, labelsize=8)
    cb.ax.xaxis.set_major_formatter(lambda v, _: fmt.format(v))
    p.bg.text((W - 3.3) / W, 1.165 / p.h, label, fontsize=8, color=MUTED2, va="center")
