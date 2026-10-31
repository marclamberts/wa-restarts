"""wa_restarts: IMPECT event data -> Waltzing Analytics set-piece and restart visuals and CSVs.

    import wa_restarts as wr
    wr.configure(json_root="events_json", out_root="wa_restarts_output", style="fog",
                 author="Marc Lamberts", brand="Waltzing Analytics", data="IMPECT", models_by="Waltzing Analytics")
    wr.leagues()                                         # league-seasons with event JSON
    wr.teams("2. Bundesliga 2026-2027")                  # clubs in one league-season
    wr.league("2. Bundesliga 2026-2027", focus_team="Hertha BSC")   # league folder (charts, visual pack, CSVs)
    wr.club("2. Bundesliga 2026-2027", "Hertha")         # one club's set-piece pack
    wr.xdanger_players("2. Bundesliga 2026-2027")        # xDanger per player (also _actions, _teams; xdanger() writes all)
"""
from pathlib import Path

from . import config
from .config import configure

__all__ = ["configure", "styles", "leagues", "teams", "score", "league", "club", "build", "xdanger", "xdanger_actions",
           "xdanger_players", "xdanger_teams"]


def styles():
    """Colour themes for configure(style=...)."""
    from .style import THEMES
    return list(THEMES)


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


def build(comp, teams=(), focus_team=None, out_root=None, refresh=False, xdanger_tables=False):
    """League folder plus a club pack for each team in teams (and the xDanger tables when xdanger_tables=True)."""
    out = [league(comp, focus_team=focus_team or (teams[0] if teams else None), out_root=out_root, refresh=refresh)]
    for t in teams:
        out.append(club(comp, t, out_root=out_root))
    if xdanger_tables:
        out.append(xdanger(comp, out_root=out_root))
    return out


def xdanger(comp, out_root=None, refresh=False):
    """<out_root>/<comp>/xdanger/: xdanger_actions.parquet, player_xdanger.csv, team_xdanger.csv."""
    from . import xdanger_tables as xd
    return xd.run(comp, out_root=out_root, refresh=refresh)


def xdanger_actions(comp, refresh=False):
    """DataFrame: every on-ball action with xG, xDanger, xDanger added / won / denied, receiver values and xShot."""
    from . import xdanger_tables as xd
    return xd.actions(comp, refresh=refresh)


def xdanger_players(comp, min_minutes=0):
    """DataFrame: xDanger per player, totals and per 90, sorted by xDanger added per 90."""
    from . import xdanger_tables as xd
    return xd.players(comp, min_minutes=min_minutes)


def xdanger_teams(comp):
    """DataFrame: xDanger and xG per match for and against, per team."""
    from . import xdanger_tables as xd
    return xd.teams(comp)


def _match_team(comp, team):
    names = teams(comp)
    if team in names:
        return team
    hit = [n for n in names if team.lower() in n.lower()]
    if len(hit) == 1:
        return hit[0]
    raise ValueError(f"Team '{team}' not in {comp}. Clubs: {names}")
