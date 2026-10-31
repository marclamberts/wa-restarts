"""Paths and branding. Each one can be set with an environment variable or at runtime with configure()."""
import os
from pathlib import Path

PKG = Path(__file__).resolve().parent


def _p(env, default):
    v = os.getenv(env)
    return Path(v).expanduser() if v else default


JSON_ROOT = _p("WA_RESTARTS_JSON", Path.cwd() / "events_json")      # <league-season>/<date>_<home>_vs_<away>_<id>.json
META_ROOT = _p("WA_RESTARTS_META", None)                            # <league-season>/squads.json, matches.json
REFERENCE = _p("WA_RESTARTS_REFERENCE", PKG / "reference")          # saved models + all-league reference tables
CACHE = _p("WA_RESTARTS_CACHE", Path.home() / ".cache" / "wa_restarts")  # per league-season scored parquet files
OUT_ROOT = _p("WA_RESTARTS_OUT", Path.cwd() / "wa_restarts_output")
XDANGER = _p("WA_RESTARTS_XDANGER", None)                           # xDanger project, only used by build_reference

# chart header and footer
AUTHOR = os.getenv("WA_RESTARTS_AUTHOR", "Marc Lamberts")             # "made by": header and footer
BRAND = os.getenv("WA_RESTARTS_BRAND", "Waltzing Analytics")          # wordmark in the header
MARK = os.getenv("WA_RESTARTS_MARK", "WA")                            # short mark in the header
DATA = os.getenv("WA_RESTARTS_DATA", "IMPECT")                        # data credit in the source line
MODELS_BY = os.getenv("WA_RESTARTS_MODELS_BY", "Waltzing Analytics")  # model credit in the source line
STYLE = os.getenv("WA_RESTARTS_STYLE", "navy")                        # colour theme, see style.THEMES
FONT = os.getenv("WA_RESTARTS_FONT", "house")                         # font pairing, see style.FONTS

# Most recent seasons were held out when the models were trained.
TEST_SEASONS = ("2026-2027", "2026")
WINDOW_S = 15.0

PATHS = ("JSON_ROOT", "META_ROOT", "REFERENCE", "CACHE", "OUT_ROOT", "XDANGER")
TEXT = ("BRAND", "MARK", "AUTHOR", "DATA", "MODELS_BY")


def configure(**kw):
    """configure(json_root=..., out_root=..., style="fog", font="plex", author=..., data=..., models_by=...)."""
    g = globals()
    for k, v in kw.items():
        name = k.upper()
        if name == "STYLE":
            from . import style
            style.use(v)
            g[name] = style.THEME
        elif name == "FONT":
            from . import style
            style.use_font(v)
            g[name] = style.FONT
        elif name in PATHS:
            g[name] = Path(v).expanduser()
        elif name in TEXT:
            g[name] = v
        else:
            raise KeyError(k)


def source(league=None, what="xDanger, xShot, xG, xGOT and restart pillars"):
    """Footer source line: data credit (and league), then what the models are and who made them."""
    data = f"Data: {DATA}, {league}" if league else f"Data: {DATA}"
    return f"{data}  ·  {what}: {MODELS_BY}" if MODELS_BY else data


def meta_root():
    """events_meta next to the JSON folder unless set."""
    return META_ROOT or JSON_ROOT.parent / "events_meta"


class Workspace:
    """Cached files for one league-season: <cache>/<league-season>/<stage>.parquet."""
    STAGES = ("events", "features", "event_labels", "scored", "xshot", "shots", "possessions", "four_models", "curves")

    def __init__(self, comp):
        self.comp = comp
        self.dir = CACHE / comp
        for s in self.STAGES:
            setattr(self, s, self.dir / f"{s}.parquet")

    @property
    def json_dir(self):
        return JSON_ROOT / self.comp

    @property
    def meta_dir(self):
        return meta_root() / self.comp

    def ready(self):
        return all(getattr(self, s).exists() for s in self.STAGES)


def model(name):
    return REFERENCE / "models" / f"{name}.joblib"


def table(name):
    return REFERENCE / name
