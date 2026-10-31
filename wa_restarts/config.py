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
BRAND = os.getenv("WA_RESTARTS_BRAND", "WALTZING ANALYTICS")
MARK = os.getenv("WA_RESTARTS_MARK", "WA")
AUTHOR = os.getenv("WA_RESTARTS_AUTHOR", "Marc Lamberts")

# Most recent seasons were held out when the models were trained.
TEST_SEASONS = ("2026-2027", "2026")
WINDOW_S = 15.0

PATHS = ("JSON_ROOT", "META_ROOT", "REFERENCE", "CACHE", "OUT_ROOT", "XDANGER")
TEXT = ("BRAND", "MARK", "AUTHOR")


def configure(**kw):
    """configure(json_root=..., out_root=..., cache=..., author=...) changes settings for this session."""
    g = globals()
    for k, v in kw.items():
        name = k.upper()
        if name in PATHS:
            g[name] = Path(v).expanduser()
        elif name in TEXT:
            g[name] = v
        else:
            raise KeyError(k)


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
