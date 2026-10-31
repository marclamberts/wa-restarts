"""Set-piece report and restart report, for one team or the whole league.

Each report opens with a cover page of headline numbers and then collects the package's own charts into one numbered PDF:
the club set-piece pack (team reports), the league visual pack and the restart-option charts (league reports).

  set piece: corners, free kicks, long throws, penalties, set-piece shots, takers, defending set pieces
  restart:   every restart as a play: options per restart type, goal kicks, throw-ins, kick-offs, sequences, pillars
"""
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .. import club_pack, config, league_visuals, options, style
from . import draw as D
from .base import Report, slug
from .data import league

TEAM_PAGES = {
    "set_piece": ["summary_cards", "xg_by_type", "goals_by_type", "volume_by_type", "type_profile_table", "type_profile_table_def",
                  "match_by_match", "cumulative", "pizza", "league_attack_defence", "league_xg_goals", "rank_xg", "rank_xg_against",
                  "corners_attack_map", "corner_zones", "corner_options", "short_corner_options", "corner_side_swing",
                  "corner_takers", "first_touch_map", "corner_zone_first_touch", "league_corner_volume_quality",
                  "league_corner_targeting", "league_first_touch", "league_second_ball", "corners_defend_map", "corner_zones_def",
                  "corner_options_def", "first_touch_map_def", "fk_attack_map", "fk_options", "fk_takers", "dfk_map", "fk_defend_map",
                  "throw_map", "penalties", "shots_for_map", "set_piece_shooters", "shots_against_map", "set_piece_shooters_against",
                  "shot_timing", "by_minute", "home_away", "opponents_table", "sp_share"],
    "restart": ["all_restart_locations", "danger_heat", "pillars", "peak_xshot_dist", "setup_time", "sequences", "throw_options",
                "throw_retention_map", "throw_takers", "goal_kick_map", "goal_kick_options", "kick_off_options", "fk_options",
                "corner_options", "short_corner_options"],
}
LEAGUE_PAGES = {
    "set_piece": ["O_team_profiles", "B01_", "B02_", "B03_", "B04_", "B06_", "B07_", "B08_", "B09_", "B10_", "B11_", "B12_", "B13_",
                  "B14_", "B15_", "B16_", "B17_", "B18_", "B19_", "B20_", "B21_", "B23_", "B24_", "O_corner", "O_short_corner",
                  "O_indirect_free_kick", "O_direct_free_kick", "O_penalty"],
    "restart": ["O_00_overview", "B22_", "O_throw_in", "O_goal_kick", "O_kick_off", "B05_", "O_corner", "O_short_corner",
                "O_indirect_free_kick", "O_direct_free_kick", "O_penalty", "O_team_profiles"],
}
TITLE = {"set_piece": "Set-piece report", "restart": "Restart report"}


def _cover_team(R, kind, c, L):
    t = c.tab
    me = t.loc[c.team]
    rank = lambda k, low=False: int((t[k] < me[k]).sum() + 1) if low else int((t[k] > me[k]).sum() + 1)
    n = len(t)
    if kind == "set_piece":
        title = f"Set pieces: {c.team}"
        sub = (f"{c.league}  ·  {int(me.matches)} matches  ·  attacking restarts (corners, opponent-half free kicks, final-third "
               "throw-ins, penalties) and the same for opponents")
        cards = [("Set-piece xG / match", f"{me.sp_xg_pm:.2f}", f"rank {rank('sp_xg_pm')} of {n}"),
                 ("Set-piece goals", f"{int(me.sp_goals)}", f"conceded {int(me.sp_goals_against)}"),
                 ("xG conceded / match", f"{me.sp_xg_against_pm:.2f}", f"rank {rank('sp_xg_against_pm', low=True)} of {n}"),
                 ("xG per corner", f"{me.corner_xg:.3f}", f"rank {rank('corner_xg')} of {n}"),
                 ("Reach danger", f"{me.reach_danger:.0%}", "attacking restarts with peak xDanger 5%+"),
                 ("Second balls", f"{me.second_ball:.0%}", f"rank {rank('second_ball')} of {n}")]
    else:
        title = f"Restarts: {c.team}"
        sub = f"{c.league}  ·  {int(me.matches)} matches  ·  every restart as a play: what it became in 30 s of live play"
        a = c.A
        cards = [("Restarts", f"{len(a):,}", f"{len(a) / me.matches:.0f} per match"),
                 ("Retention", f"{a.retention.mean():.0%}", f"league {c.d.retention.mean():.0%}"),
                 ("Reach danger", f"{a.reach_danger.mean():.0%}", f"league {c.d.reach_danger.mean():.0%}"),
                 ("Another set piece", f"{a.set_piece.mean():.0%}", f"league {c.d.set_piece.mean():.0%}"),
                 ("Goal-kick retention", f"{me.gk_retention:.0%}", f"rank {rank('gk_retention')} of {n}"),
                 ("Short goal kicks", f"{me.gk_short:.0%}", f"league {t.gk_short.mean():.0%}")]
    p = R.page(title, sub)
    p.cards(cards[:3], top=1.85, height=1.0)
    p.cards(cards[3:], top=3.0, height=1.0)
    p.section("By restart type: theirs against the league average", top=4.35)
    ax = p.axes(top=4.6, bottom=0.95)
    d = c.A if kind == "restart" else c.A[c.A.attacking]
    lg = c.d if kind == "restart" else c.d[c.d.attacking]
    order = [r for r in ["Corner", "Short corner", "Indirect free kick", "Direct free kick", "Penalty", "Throw-in", "Goal kick",
                         "Kick-off"] if (d.restart == r).any()]
    rows = []
    for r in order:
        g, l = d[d.restart == r], lg[lg.restart == r]
        rows.append([r, len(g), g.xg_for.mean(), l.xg_for.mean(), g.reach_danger.mean(), l.reach_danger.mean(), g.retention.mean(),
                     l.retention.mean(), int(g.goals_for.sum())])
    cols = [[None, None] + [((style.ACCENT if r[k] >= r[k + 1] else style.BLUE), 0.35) if k in (2, 4, 6) and np.isfinite(r[k])
                            and np.isfinite(r[k + 1]) else None for k in range(2, 9)] for r in rows]
    D.table(ax, ["Restart", "N", "xG", "League", "Danger", "League", "Kept", "League", "Goals"], rows,
            [0.24, 0.08, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.08], ["{}", "{:.0f}", "{:.3f}", "{:.3f}", "{:.0%}", "{:.0%}", "{:.0%}", "{:.0%}",
                                                             "{:.0f}"], cols)


def _cover_league(R, kind, L):
    r = L.restarts
    tm = L.team_match
    if kind == "set_piece":
        a = r[r.attacking]
        title = f"Set pieces: {L.comp}"
        sub = f"{len(L.matches)} matches  ·  {len(a):,} attacking restarts  ·  league-wide set-piece report"
        top = tm.groupby("team").sp_xg.mean().sort_values(ascending=False)
        cards = [("Set-piece xG / match", f"{tm.sp_xg.mean():.2f}", "per team"), ("Set-piece goals", f"{int(a.goals_for.sum())}",
                 f"{a.goals_for.sum() / max(a.xg_for.sum(), 1e-9):.2f} per xG"), ("Best attack", top.index[0], f"{top.iloc[0]:.2f} xG / match"),
                 ("xG per corner", f"{r[r.restart == 'Corner'].xg_for.mean():.3f}", f"{(r.restart == 'Corner').sum():,} corners"),
                 ("Reach danger", f"{a.reach_danger.mean():.0%}", "attacking restarts with peak xDanger 5%+"),
                 ("Second balls", f"{a.second_ball.mean():.0%}", "won by the taking team")]
    else:
        title = f"Restarts: {L.comp}"
        sub = f"{len(L.matches)} matches  ·  {len(r):,} restarts  ·  league-wide restart report"
        gk = r[r.restart == "Goal kick"]
        cards = [("Restarts / match", f"{len(r) / len(L.matches):.0f}", "both teams"), ("Retention", f"{r.retention.mean():.0%}", "all restarts"),
                 ("Reach danger", f"{r.reach_danger.mean():.0%}", "all restarts"), ("Goal kicks played short",
                 f"{gk.option.str.startswith('Short').mean():.0%}", f"retention {gk.retention.mean():.0%}"),
                 ("Throw-ins / match", f"{(r.restart == 'Throw-in').sum() / len(L.matches):.0f}",
                  f"retention {r[r.restart == 'Throw-in'].retention.mean():.0%}"),
                 ("Another set piece", f"{r.set_piece.mean():.0%}", "won after the restart")]
    p = R.page(title, sub)
    p.cards(cards[:3], top=1.85, height=1.0)
    p.cards(cards[3:], top=3.0, height=1.0)
    p.section("By restart type", top=4.35)
    ax = p.axes(top=4.6, bottom=0.95)
    d = r if kind == "restart" else r[r.attacking]
    order = [x for x in ["Corner", "Short corner", "Indirect free kick", "Direct free kick", "Penalty", "Throw-in", "Goal kick",
                         "Kick-off"] if (d.restart == x).any()]
    rows = [[x, (d.restart == x).sum(), d[d.restart == x].xg_for.mean(), d[d.restart == x].reach_danger.mean(),
             d[d.restart == x].retention.mean(), d[d.restart == x].set_piece.mean(), int(d[d.restart == x].goals_for.sum())] for x in order]
    D.table(ax, ["Restart", "N", "xG", "Danger", "Kept", "Another SP", "Goals"], rows, [0.3, 0.1, 0.12, 0.12, 0.12, 0.12, 0.12],
            ["{}", "{:.0f}", "{:.3f}", "{:.0%}", "{:.0%}", "{:.0%}", "{:.0f}"])


def build(comp, kind, team=None, out_root=None):
    """kind: 'set_piece' or 'restart'. team=None builds the league-wide report."""
    if kind not in TITLE:
        raise ValueError("kind must be 'set_piece' or 'restart'")
    L = league(comp)
    folder = Path(out_root or config.OUT_ROOT) / comp / "reports" / kind.replace("_", "-")
    tag = f"{style.THEME}_{style.FONT}"
    if team:
        c = club_pack.Ctx(comp, team)
        R = Report(kind, comp, f"{c.team}  ·  {TITLE[kind]}  ·  {comp}")
        _cover_team(R, kind, c, L)
        fns = {f.__name__: f for f in club_pack.VIS}
        for name in TEAM_PAGES[kind]:
            try:
                R.pages.append(fns[name](c))
            except Exception as e:  # noqa: BLE001
                print(f"  skipped {name}: {type(e).__name__}: {e}", flush=True)
                plt.close("all")
        return R.save(folder, f"{slug(c.team)}_{tag}")
    R = Report(kind, comp, f"{comp}  ·  {TITLE[kind]}")
    _cover_league(R, kind, L)
    with tempfile.TemporaryDirectory() as tmp, style.collect() as got:
        options.run(comp, tmp)
        league_visuals.run(comp, Path(tmp) / "visuals")
    by_name = {}
    for p, name in got:
        by_name.setdefault(name, p)
    used = set()
    for key in LEAGUE_PAGES[kind]:
        for name, p in by_name.items():
            if name.startswith(key) and name not in used and (key.endswith("_") or name == key):
                R.pages.append(p)
                used.add(name)
    for name, p in by_name.items():
        if name not in used:
            plt.close(p.fig)
    return R.save(folder, f"league_{tag}")
