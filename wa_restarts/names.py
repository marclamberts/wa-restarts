"""Club names and home sides per league-season.

IMPECT's squads.json / matches.json in events_meta/<league-season> give the full names (with accents) and the home
squad. Without them, names are resolved from the match file names as in xDanger 04_outputs.py (accents are lost there).
"""
import json
import re
from collections import defaultdict

import pandas as pd

from .config import Workspace

FILE = re.compile(r"\d{4}-\d{2}-\d{2}_(.+)_vs_(.+)_(\d+)\.json$")
# match file names lose their non-ASCII letters; repair the common ones (xDanger player_pack.py)
FIX = {"M nchengladbach": "Mönchengladbach", "M nchen": "München", "K ln": "Köln", "W rzburg": "Würzburg",
       "D sseldorf": "Düsseldorf", "F rth": "Fürth", "N rnberg": "Nürnberg", "Saarbr cken": "Saarbrücken",
       "Osnabr ck": "Osnabrück", "M nster": "Münster", "L beck": "Lübeck", "G teborg": "Göteborg", "H cken": "Häcken",
       "L ttich": "Lüttich", "Lillestr m": "Lillestrøm", "V lerenga": "Vålerenga", "Reykjav k": "Reykjavík",
       "K pavog": "Kópavog", "Alav s": "Alavés", "Legan s": "Leganés", "V ster s": "Västerås", "V rnamo": "Värnamo",
       "K ge": "Køge", "Hiller d": "Hillerød", "Keflav k": "Keflavík", "V kingur": "Víkingur", "Th r ": "Þór "}


def fix(name):
    for a, b in FIX.items():
        name = name.replace(a, b)
    return name


def _file_pairs(ws):
    pairs = {}
    for f in ws.json_dir.glob("*.json"):
        m = FILE.match(f.name)
        if m:
            pairs[int(m.group(3))] = (fix(m.group(1).replace("_", " ")), fix(m.group(2).replace("_", " ")))
    return pairs


def squad_names(comp, events=None):
    """DataFrame squad_id, team for one league-season."""
    ws = Workspace(comp)
    f = ws.meta_dir / "squads.json"
    if f.exists():
        sq = json.loads(f.read_text())
        return pd.DataFrame({"squad_id": [s["id"] for s in sq], "team": [s["name"] for s in sq]})
    # squad ids per match intersected over every match a file-name club appears in
    events = pd.read_parquet(ws.events, columns=["match_id", "squad_id"]) if events is None else events
    names = _file_pairs(ws)
    cand = defaultdict(list)
    for mid, sq in events.groupby("match_id")["squad_id"].unique().items():
        for n in names.get(mid, ()):
            cand[n].append(set(sq))
    out = {}
    for n, sets in cand.items():
        common = set.intersection(*sets)
        if len(common) == 1:
            out[common.pop()] = n
    for n, sets in cand.items():
        if n not in out.values():
            left = set.intersection(*sets) - set(out)
            if len(left) == 1:
                out[left.pop()] = n
    return pd.DataFrame({"squad_id": list(out), "team": list(out.values())})


def home_squads(comp):
    """{match_id: home squad_id}, or {} when matches.json is missing."""
    f = Workspace(comp).meta_dir / "matches.json"
    if not f.exists():
        return {}
    return {m["id"]: m["homeSquadId"] for m in json.loads(f.read_text())}


def home_from_files(comp):
    """{match_id: home club as written in the file name}."""
    return {mid: h for mid, (h, _) in _file_pairs(Workspace(comp)).items()}
