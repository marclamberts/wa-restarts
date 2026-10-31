"""wa_restarts: IMPECT event data -> Waltzing Analytics set-piece and restart visuals and CSVs.

    import wa_restarts as wr
    wr.configure(json_root="events_json", out_root="wa_restarts_output")
    wr.leagues()                                         # league-seasons with event JSON
    wr.teams("2. Bundesliga 2026-2027")                  # clubs in one league-season
    wr.league("2. Bundesliga 2026-2027", focus_team="Hertha BSC")   # league folder (charts, visual pack, CSVs)
    wr.club("2. Bundesliga 2026-2027", "Hertha")         # one club's set-piece pack
"""
from pathlib import Path

from . import config
from .config import configure

__all__ = ["configure", "leagues", "teams", "score", "league", "club", "build"]


def leagues():
    return sorted(p.name for p in config.JSON_ROOT.iterdir() if p.is_dir() and any(p.glob("*.json")))


def score(comp, refresh=False, workers=8):
    """Score a league-season into the cache (skipped when cached, unless refresh=True or new match files arrived)."""
    from .scoring import refresh as rf
    ws = config.Workspace(comp)
    if not refresh and ws.ready():
        newest = max((f.stat().st_mtime for f in ws.json_dir.glob("*.json")), default=0)
        if newest <= ws.four_models.stat().st_mtime:
            return ws
        print("new match files since the last scoring run: rescoring", flush=True)
    return rf.run(comp, workers=workers)


def teams(comp):
    from .names import squad_names
    ws = config.Workspace(comp)
    if not (ws.meta_dir / "squads.json").exists() and not ws.events.exists():
        score(comp)  # names come from the match files, which need the extracted events
    return sorted(squad_names(comp).team)


def league(comp, focus_team=None, out_root=None, refresh=False):
    """<out_root>/<comp>/: restart_options.csv, team_set_piece_profiles.csv, charts/O_*.png, visual pack/B*.png + CSVs."""
    from . import league_visuals, options
    score(comp, refresh=refresh)
    folder = Path(out_root or config.OUT_ROOT) / comp
    options.run(comp, folder)
    league_visuals.run(comp, folder / "visual pack", focus_team=_match_team(comp, focus_team) if focus_team else None)
    print(f"league deliverables in {folder}", flush=True)
    return folder


def club(comp, team, out_root=None, refresh=False):
    """<out_root>/<comp>/<team> set pieces/: the club's NN_*.png pack and restarts_for / against CSVs."""
    from . import club_pack
    score(comp, refresh=refresh)
    return club_pack.run(comp, team, out_root=out_root)


def build(comp, teams=(), focus_team=None, out_root=None, refresh=False):
    """League folder plus a club pack for each team in teams."""
    out = [league(comp, focus_team=focus_team or (teams[0] if teams else None), out_root=out_root, refresh=refresh)]
    for t in teams:
        out.append(club(comp, t, out_root=out_root))
    return out


def _match_team(comp, team):
    names = teams(comp)
    if team in names:
        return team
    hit = [n for n in names if team.lower() in n.lower()]
    if len(hit) == 1:
        return hit[0]
    raise ValueError(f"Team '{team}' not in {comp}. Clubs: {names}")
