"""Report frame: pages in the current style and font, numbered "nn / NN", written as one PDF plus one PNG per page.

Landscape pages are 11 x 7.78 in and portrait pages 7.78 x 11 in (A4 proportions, same width as the chart pages).
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .. import config, style

LAND = (11.0, 7.78)
PORT = (7.78, 11.0)


def slug(s):
    return re.sub(r"[^\w\-. ]+", "", str(s)).strip().replace(" ", "_")


class Report:
    def __init__(self, kind, league, eyebrow):
        self.kind, self.league, self.eyebrow = kind, league, eyebrow
        self.pages = []

    def page(self, title, subtitle, portrait=False, source=None):
        w, h = PORT if portrait else LAND
        p = style.Page(h, self.eyebrow, title, subtitle, source=source or config.source(self.league), w=w)
        self.pages.append(p)
        return p

    def add(self, pages):
        """Add pages produced by chart functions (style.collect)."""
        for p, _ in pages:
            self.pages.append(p)

    def chart(self, fn, *args, **kw):
        """Run a chart function that saves its page(s) and add them; a chart that cannot be drawn is skipped."""
        with style.collect() as got:
            try:
                fn(*args, **kw)
            except Exception as e:  # noqa: BLE001
                print(f"  skipped {getattr(fn, '__name__', fn)}: {type(e).__name__}: {e}", flush=True)
        self.add(got)

    def save(self, folder, name, png=True):
        """<folder>/<name>.pdf and <folder>/<name>/NN.png."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        n = len(self.pages)
        pdf = folder / f"{name}.pdf"
        png_dir = folder / name
        if png:
            png_dir.mkdir(exist_ok=True)
            for old in png_dir.glob("*.png"):
                old.unlink()
        with PdfPages(pdf) as out:
            for i, p in enumerate(self.pages, 1):
                if n > 1:  # page number in front of the author line, as in the match reports
                    for t in p.bg.texts:
                        if t.get_text().startswith(f"{config.AUTHOR} | Created"):
                            t.set_text(f"{i:02d} / {n:02d}      {t.get_text()}")
                out.savefig(p.fig, facecolor=style.BG)
                if png:
                    p.fig.savefig(png_dir / f"{i:02d}.png", dpi=200, facecolor=style.BG)
                plt.close(p.fig)
        self.pages = []
        print(f"{n} pages -> {pdf}", flush=True)
        return pdf
