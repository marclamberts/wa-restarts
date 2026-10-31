"""Club set-piece pack: 50+ WA navy PNGs on one club's restarts in one league-season, attacking and defending.

Ported from xDanger team_setpiece_pack.py.

    from wa_restarts import club_pack
    club_pack.teams("League One 2026-2027")                 # club names in that league-season
    club_pack.run("League One 2026-2027", "FC Barnsley")

Every restart is scored over the same 30 s live window as 31_restart_four_models.py (xG, xGOT, goals, peak xDanger and
xShot for and against, retention, dominance, second balls, another set piece won) and classified into options by
33_restart_options.py. Defending = the opponents' restarts in this club's matches. Coordinates are always from the
side taking the restart, attacking upwards / to the right.
Writes <out_root>/<league>/<team> set pieces/NN_<name>.png and CSVs. A visual that cannot be drawn is skipped and listed."""
import difflib
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mplsoccer import Pitch, VerticalPitch

from . import config
from . import options as ro
from .names import home_from_files, home_squads, squad_names
from .style import (ACCENT, BG, BLUE, CARD, DANGER_CMAP, GOLD, GREY, HAIR, INK, MUTED, MUTED2, RED, TEAL, W, L, Page,
                      clean)

SRC = "Data: IMPECT  ·  restart four-model windows (30 s) and options: Waltzing Analytics"
PITCH = dict(pitch_type="custom", pitch_length=105, pitch_width=68, pitch_color=BG, line_color=HAIR, linewidth=1, line_zorder=2)
HEAT = LinearSegmentedColormap.from_list("wa_heat", [BG, "#2A3A48", GOLD, ACCENT])
TYPES = ["Corner", "Short corner", "Indirect free kick", "Direct free kick", "Penalty", "Throw-in"]
TCOL = {"Corner": ACCENT, "Short corner": GOLD, "Indirect free kick": TEAL, "Direct free kick": BLUE, "Penalty": "#8C7BD1",
        "Throw-in": "#6B7884", "Goal kick": MUTED2, "Kick-off": GREY}
CORNER_ZONES = ["near post", "six-yard centre", "far post", "penalty spot", "edge of the box"]


def teams(league):
    return sorted(squad_names(league).team)


# ---------------------------------------------------------------- data
class Ctx:
    def __init__(self, league, team):
        self.league = league
        names = teams(league)
        if team not in names:
            hit = [n for n in names if team.lower() in n.lower()]
            if len(hit) == 1:
                team = hit[0]
            else:
                raise ValueError(f"Team '{team}' not in {league}. Did you mean: {hit or difflib.get_close_matches(team, names, 5, 0.3)}")
        self.team = team
        d = ro.classify(ro.load(league))
        self.ws = config.Workspace(league)
        sm = squad_names(league)
        d = d.merge(sm[["squad_id", "team"]], on="squad_id", how="left")
        ev = pd.read_parquet(self.ws.events, columns=["match_id", "idx", "date", "period", "t", "player_name"])
        d = d.merge(ev, on=["match_id", "idx"], how="left")
        d["minute"] = (d.t - (d.period - 1) * 10000) / 60 + 45 * (d.period - 1)
        tm = d.groupby("match_id").team.unique()
        d["opponent"] = [next((x for x in tm[m] if x != t), None) for m, t in zip(d.match_id, d.team)]
        d["attacking"] = d.restart.isin(TYPES) & ~((d.restart == "Indirect free kick") & d.option.str.startswith("Own half")) \
            & ~((d.restart == "Throw-in") & d.option.str.startswith("Elsewhere"))
        d["fold_y"] = np.where(d.y >= 0, d.end_y, -d.end_y)
        d["side"] = np.where(d.y >= 0, "Left", "Right")
        d["zone"] = np.where(d.restart == "Corner", d.option.str.split(", ").str[-1], None)
        d["swing"] = np.where(d.restart == "Corner", d.option.str.split(", ").str[0], None)
        self.home = self._home_away(league, sm)
        d["home"] = [self.home.get(m) == t for m, t in zip(d.match_id, d.team)]
        self.d = d
        self.A = d[d.team == team].copy()
        self.D = d[d.opponent == team].copy()
        self.matches = sorted(d[(d.team == team) | (d.opponent == team)].match_id.unique())
        self.nm = len(self.matches)
        self.shots = self._window_shots()
        self.shA = self.shots[self.shots.r_team == team]
        self.shD = self.shots[self.shots.r_opponent == team]
        self.first = self._first_touch()
        self.tab = self._club_table()
        self.pool = self.tab

    def _home_away(self, league, sm):
        hs = home_squads(league)
        if hs:
            name = dict(zip(sm.squad_id, sm.team))
            return {m: name.get(s) for m, s in hs.items()}
        out = home_from_files(league)
        # map file-name home side to a club name: the club whose letters match best
        names = self.d_names = teams(league)
        key = lambda s: re.sub(r"[^a-z]", "", s.lower())
        res = {}
        for mid, h in out.items():
            best = max(names, key=lambda n: difflib.SequenceMatcher(None, key(n), key(h)).ratio())
            res[mid] = best
        return res

    def _window_shots(self):
        s = pd.read_parquet(self.ws.scored, columns=["match_id", "idx", "period", "t", "squad_id", "action_type",
                                                                        "action", "result", "x", "y", "xg", "player_name"])
        sh = s[s.action_type == "SHOT"].copy()
        r = self.d[["match_id", "idx", "squad_id", "restart", "option", "team", "opponent", "attacking", "period", "t"]]
        sh, r = sh.sort_values("idx"), r.sort_values("idx")
        m = pd.merge_asof(sh, r.rename(columns={"idx": "r_idx", "squad_id": "r_squad", "t": "r_t", "period": "r_period",
                                                "team": "r_team", "opponent": "r_opponent"}),
                          left_on="idx", right_on="r_idx", by="match_id", direction="backward")
        m = m[(m.r_squad == m.squad_id) & (m.r_period == m.period) & (m.t - m.r_t <= 30) & m.attacking.astype(bool)].copy()
        m["goal"] = (m.result == "SUCCESS").astype(int)
        m["delay"] = m.t - m.r_t
        return m

    def _first_touch(self):
        e = pd.read_parquet(self.ws.events, columns=["match_id", "idx", "squad_id", "action_type", "x", "y",
                                                                         "player_name"])
        e = e.sort_values(["match_id", "idx"])
        nxt = e.groupby("match_id").shift(-1)
        e = e.assign(n_squad=nxt.squad_id, n_type=nxt.action_type, n_x=nxt.x, n_y=nxt.y, n_player=nxt.player_name)
        co = self.d[self.d.restart == "Corner"][["match_id", "idx", "squad_id", "team", "opponent", "y", "end_x", "end_y", "xg_for",
                                                  "zone"]]
        f = co.merge(e[["match_id", "idx", "n_squad", "n_type", "n_x", "n_y", "n_player"]], on=["match_id", "idx"], how="left")
        f["won"] = (f.n_squad == f.squad_id).astype(float)
        return f

    def _club_table(self):
        d = self.d
        att = d[d.attacking]
        rows = {}
        for team, g in d.groupby("team"):
            nm = d[(d.team == team) | (d.opponent == team)].match_id.nunique()
            a = att[att.team == team]
            o = att[att.opponent == team]
            co = g[g.restart.isin(["Corner", "Short corner"])]
            oc = d[(d.opponent == team) & d.restart.isin(["Corner", "Short corner"])]
            th = g[(g.restart == "Throw-in") & g.option.str.startswith("Final third")]
            fk = g[(g.restart == "Indirect free kick") & g.option.str.startswith("Opp. half")]
            gk = g[g.restart == "Goal kick"]
            ft = self.first if hasattr(self, "first") else None
            rows[team] = {
                "matches": nm, "sp_xg_pm": a.xg_for.sum() / nm, "sp_xg_against_pm": o.xg_for.sum() / nm,
                "sp_goals": a.goals_for.sum(), "sp_goals_against": o.goals_for.sum(), "sp_shots_pm": a.shots_for.sum() / nm,
                "corners_pm": len(co) / nm, "corners_against_pm": len(oc) / nm, "corner_xg": co.xg_for.mean(),
                "corner_xg_against": oc.xg_for.mean(), "short_corner": (co.restart == "Short corner").mean(),
                "corner_goal_area": co.option.str.contains("six-yard|near post|far post|penalty spot").mean(),
                "inswing": co.option.str.startswith("Inswing").mean(),
                "fk_into_box": fk.option.str.contains("into box").mean(), "fk_xg": fk.xg_for.mean(),
                "long_throw": th.option.str.contains("long throw").mean(), "throw_xg": th.xg_for.mean(),
                "gk_short": gk.option.str.startswith("Short").mean(), "gk_retention": gk.retention.mean(),
                "reach_danger": a.reach_danger.mean(), "peak_xshot": a.peak_xshot.mean(), "second_ball": a.second_ball.mean(),
                "set_piece_won": a.set_piece.mean(), "retention": a.retention.mean(), "dominance": a.dominance.mean(),
                "danger_against": o.reach_danger.mean(), "goal_against_rate": a.goal_against.mean(),
                "first_touch": ft[ft.team == team].won.mean() if ft is not None else np.nan,
                "first_touch_def": 1 - ft[ft.opponent == team].won.mean() if ft is not None else np.nan}
        t = pd.DataFrame(rows).T.astype(float)
        t["sp_goal_diff"] = t.sp_goals - t.sp_goals_against
        t["sp_xg_diff_pm"] = t.sp_xg_pm - t.sp_xg_against_pm
        return t.sort_values("sp_xg_pm", ascending=False)


# ---------------------------------------------------------------- helpers
def page(c, title, sub, h=7.4):
    return Page(h, f"{c.team}  ·  set pieces  ·  {c.league}", title, sub, source=SRC)


def axes(p, top=1.75, bottom=0.95, left=1.0, right=0.6, **kw):
    return p.axes(top=top, bottom=bottom, left=left, right=right, **kw)


def px(x):
    return np.asarray(x, float) + 52.5


def py(y):
    return np.asarray(y, float) + 34


def half(p, top=1.75, bottom=1.1, left=None, width=6.0, pad_bottom=-20):
    pt = VerticalPitch(half=True, pad_bottom=pad_bottom, pad_top=2, pad_left=1, pad_right=1, **PITCH)
    left = (W - width) / 2 if left is None else left
    ax = p.axes(top=top, bottom=bottom, left=left, right=W - left - width)
    pt.draw(ax=ax)
    ax.set_anchor("N")
    return pt, ax


def full(p, top=1.75, bottom=1.0):
    pt = Pitch(pad_top=1, pad_bottom=1, pad_left=1, pad_right=1, **PITCH)
    ax = p.axes(top=top, bottom=bottom, left=L, right=L)
    pt.draw(ax=ax)
    ax.set_anchor("N")
    return pt, ax


def dot(color, label, ms=7, hollow=False, marker="o"):
    return Line2D([], [], marker=marker, ls="", ms=ms, label=label, mfc="none" if hollow else color, mec=color)


def legend(ax, h, y=-0.02, ncol=4):
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False, fontsize=8, labelcolor=MUTED2,
              handletextpad=0.4, columnspacing=1.2)


def rank_text(c, col, low_good=False):
    t = c.tab[col].rank(ascending=low_good, method="min")
    return f"{int(t[c.team])} of {len(t)}"


def table_page(c, title, sub, rows, cols, fmt, colour_by=None, h=None):
    """rows: list of (label, values...). Draws a simple coloured table."""
    h = h or 2.6 + 0.36 * len(rows)
    p = page(c, title, sub, h=h)
    ax = axes(p, top=1.95, bottom=0.8, left=L, right=L)
    ax.set_xlim(0, len(cols) + 2.2), ax.set_ylim(len(rows), -0.6), ax.axis("off")
    for j, lab in enumerate(cols):
        ax.text(2.2 + j + 0.5, -0.35, lab, ha="center", fontsize=7.5, weight="bold", color=MUTED2)
    for i, r in enumerate(rows):
        ax.text(0, i + 0.5, r[0], va="center", fontsize=8.5, color=INK)
        for j, v in enumerate(r[1:]):
            if colour_by is not None:
                cv = colour_by(i, j, v)
                if cv is not None:
                    ax.add_patch(Rectangle((2.2 + j + 0.04, i + 0.08), 0.92, 0.84, color=cv[0], alpha=cv[1], lw=0))
            ax.text(2.2 + j + 0.5, i + 0.5, "" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt[j].format(v),
                    ha="center", va="center", fontsize=8.3, color=INK)
    return p


def vs_colour(ref, invert=()):
    def f(i, j, v):
        r = ref[i][j]
        if r is None or v is None or not np.isfinite(v) or not np.isfinite(r) or r == 0:
            return None
        rel = v / r - 1
        good = -rel if j in invert else rel
        return (ACCENT if good >= 0 else BLUE), 0.12 + 0.7 * min(1, abs(good) / 0.6)
    return f


def league_bar(c, col, title, sub, fmt="{:.2f}", low_good=False):
    t = c.tab.sort_values(col, ascending=low_good)
    p = page(c, title, sub, h=2.4 + 0.27 * len(t))
    ax = axes(p, top=1.85, left=2.6, right=0.8)
    y = np.arange(len(t))[::-1]
    ax.barh(y, t[col], color=[ACCENT if i == c.team else GREY for i in t.index], height=0.7)
    for yi, v in zip(y, t[col]):
        ax.text(v, yi, "  " + fmt.format(v), va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y), ax.set_yticklabels(t.index, fontsize=8)
    clean(ax, grid="x")
    return p


def league_scatter(c, xcol, ycol, xlab, ylab, title, sub, invert_y=False, diag=False, pct=False):
    t = c.tab
    p = page(c, title, sub)
    ax = axes(p, left=1.1)
    ax.scatter(t[xcol], t[ycol], s=40, color=GREY, lw=0)
    for n, r in t.iterrows():
        ax.text(r[xcol], r[ycol], "  " + n.replace("FC ", ""), fontsize=7, color=INK if n == c.team else MUTED2, va="center",
                weight="bold" if n == c.team else "normal")
    ax.scatter(t.at[c.team, xcol], t.at[c.team, ycol], s=140, color=ACCENT, edgecolor=INK, lw=1.2, zorder=4)
    ax.axvline(t[xcol].mean(), color=HAIR, ls=(0, (3, 3))), ax.axhline(t[ycol].mean(), color=HAIR, ls=(0, (3, 3)))
    if diag:
        m = max(t[xcol].max(), t[ycol].max()) * 1.05
        ax.plot([0, m], [0, m], color=MUTED, ls=(0, (3, 3)))
    if invert_y:
        ax.invert_yaxis()
    if pct:
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel(xlab), ax.set_ylabel(ylab)
    clean(ax, grid="both")
    return p


VIS = []


def viz(fn):
    VIS.append(fn)
    return fn


# ---------------------------------------------------------------- overview
@viz
def summary_cards(c):
    t = c.tab.loc[c.team]
    a, o = c.A[c.A.attacking], c.D[c.D.attacking]
    p = page(c, f"{c.team}: set pieces in numbers", f"{c.nm} matches  ·  attacking restarts (corners, free kicks in the opponent "
             "half, final-third throw-ins, penalties) and the same for opponents", h=6.0)
    rows = [[("Set-piece xG", f"{a.xg_for.sum():.1f}", f"{t.sp_xg_pm:.2f} per match  ·  rank {rank_text(c, 'sp_xg_pm')}"),
             ("Set-piece goals", f"{int(a.goals_for.sum())}", f"{int(a.shots_for.sum())} shots"),
             ("xG conceded", f"{o.xg_for.sum():.1f}", f"{t.sp_xg_against_pm:.2f} per match  ·  rank {rank_text(c, 'sp_xg_against_pm', True)}"),
             ("Goals conceded", f"{int(o.goals_for.sum())}", f"{int(o.shots_for.sum())} shots"),
             ("xG difference", f"{a.xg_for.sum() - o.xg_for.sum():+.1f}", f"rank {rank_text(c, 'sp_xg_diff_pm')}")],
            [("Corners", f"{int((c.A.restart.isin(['Corner', 'Short corner'])).sum())}", f"{t.corners_pm:.1f} per match"),
             ("xG per corner", f"{t.corner_xg:.3f}", f"rank {rank_text(c, 'corner_xg')}"),
             ("Reach danger", f"{t.reach_danger:.0%}", "attacking restarts with peak xDanger 5%+"),
             ("Second balls", f"{t.second_ball:.0%}", "won after the first contact"),
             ("First touch, corners", f"{t.first_touch:.0%}", f"defending: {t.first_touch_def:.0%}")]]
    for i, r in enumerate(rows):
        p.cards(r, top=1.75 + i * 1.2, height=1.0)
    return p


def by_match(c):
    a = c.A[c.A.attacking].groupby("match_id").agg(xg=("xg_for", "sum"), g=("goals_for", "sum"))
    o = c.D[c.D.attacking].groupby("match_id").agg(xg=("xg_for", "sum"), g=("goals_for", "sum"))
    m = pd.DataFrame(index=c.matches).join(a).join(o, rsuffix="_ag").fillna(0)
    info = c.d.drop_duplicates("match_id").set_index("match_id")
    m["date"] = info.date
    m["opp"] = [c.d[(c.d.match_id == x) & (c.d.team != c.team)].team.iloc[0] if ((c.d.match_id == x) & (c.d.team != c.team)).any()
                else "?" for x in m.index]
    m["ha"] = ["H" if c.home.get(x) == c.team else "A" for x in m.index]
    return m.sort_values("date")


@viz
def match_by_match(c):
    m = by_match(c)
    p = page(c, f"{c.team}: set-piece xG, match by match", "Up = created, down = conceded  ·  dots = goals", h=7.8)
    ax = axes(p, bottom=2.0)
    x = np.arange(len(m))
    ax.bar(x, m.xg, color=ACCENT, width=0.7)
    ax.bar(x, -m.xg_ag, color=BLUE, width=0.7)
    for xi, g, ga in zip(x, m.g, m.g_ag):
        for k in range(int(g)):
            ax.scatter(xi, m.xg.iloc[xi] + 0.06 + 0.08 * k, s=22, color=INK, zorder=3)
        for k in range(int(ga)):
            ax.scatter(xi, -m.xg_ag.iloc[xi] - 0.06 - 0.08 * k, s=22, color=INK, zorder=3)
    ax.axhline(0, color=HAIR)
    ax.set_xticks(x), ax.set_xticklabels([f"{pd.to_datetime(d):%d %b} {h} {o.replace('FC ', '')[:14]}" for d, h, o in
                                          zip(m.date, m.ha, m.opp)], rotation=90, fontsize=6.8)
    clean(ax, grid="y")
    return p


@viz
def cumulative(c):
    m = by_match(c)
    p = page(c, f"{c.team}: the set-piece season", "Cumulative set-piece xG and goals, for and against")
    ax = axes(p, right=1.5)
    x = np.arange(1, len(m) + 1)
    for col, lab, colr, ls in (("xg", "xG for", ACCENT, "-"), ("g", "Goals for", ACCENT, (0, (3, 2))),
                               ("xg_ag", "xG against", BLUE, "-"), ("g_ag", "Goals against", BLUE, (0, (3, 2)))):
        ax.plot(x, m[col].cumsum(), color=colr, ls=ls, lw=2)
        ax.text(x[-1] + 0.2, m[col].sum(), f"{lab} {m[col].sum():.1f}", color=colr, fontsize=8, va="center")
    ax.set_xlabel("Match"), ax.set_xlim(0.5, len(m) + 0.5)
    clean(ax, grid="y")
    return p


def type_split(c, col):
    a = c.A[c.A.attacking].groupby("restart")[col].sum().reindex(TYPES, fill_value=0)
    o = c.D[c.D.attacking].groupby("restart")[col].sum().reindex(TYPES, fill_value=0)
    lg = c.d[c.d.attacking].groupby("restart")[col].sum().reindex(TYPES, fill_value=0) / c.d.team.nunique()
    return a, o, lg


@viz
def xg_by_type(c):
    a, o, lg = type_split(c, "xg_for")
    p = page(c, f"{c.team}: where the set-piece xG comes from", "Season xG by restart type: created, conceded, league average club")
    ax = axes(p, left=2.0)
    y = np.arange(len(TYPES))[::-1]
    ax.barh(y + 0.27, a, height=0.26, color=ACCENT, label="Created")
    ax.barh(y, o, height=0.26, color=BLUE, label="Conceded")
    ax.barh(y - 0.27, lg, height=0.26, color=GREY, label="League average")
    ax.set_yticks(y), ax.set_yticklabels(TYPES)
    ax.legend(frameon=False, labelcolor=MUTED2, fontsize=8)
    clean(ax, grid="x")
    return p


@viz
def goals_by_type(c):
    a, o, lg = type_split(c, "goals_for")
    p = page(c, f"{c.team}: set-piece goals by type", "Goals within 30 s of the restart: scored, conceded, league average club")
    ax = axes(p, left=2.0)
    y = np.arange(len(TYPES))[::-1]
    ax.barh(y + 0.27, a, height=0.26, color=ACCENT, label="Scored")
    ax.barh(y, o, height=0.26, color=BLUE, label="Conceded")
    ax.barh(y - 0.27, lg, height=0.26, color=GREY, label="League average")
    ax.set_yticks(y), ax.set_yticklabels(TYPES)
    ax.legend(frameon=False, labelcolor=MUTED2, fontsize=8)
    clean(ax, grid="x")
    return p


@viz
def volume_by_type(c):
    a = c.A.groupby("restart").size().reindex(list(TCOL), fill_value=0) / c.nm
    o = c.D.groupby("restart").size().reindex(list(TCOL), fill_value=0) / c.nm
    lg = c.d.groupby("restart").size().reindex(list(TCOL), fill_value=0) / c.d.groupby("team").match_id.nunique().sum()
    p = page(c, f"{c.team}: how many restarts, of what kind", "Per match: taken by them, by their opponents, league average")
    ax = axes(p, left=2.0)
    y = np.arange(len(a))[::-1]
    ax.barh(y + 0.27, a, height=0.26, color=ACCENT, label=c.team)
    ax.barh(y, o, height=0.26, color=BLUE, label="Opponents")
    ax.barh(y - 0.27, lg, height=0.26, color=GREY, label="League average")
    ax.set_yticks(y), ax.set_yticklabels(a.index)
    ax.legend(frameon=False, labelcolor=MUTED2, fontsize=8)
    clean(ax, grid="x")
    return p


@viz
def type_profile_table(c):
    cols = ["Reach\ndanger", "Peak\nxShot", "xG per\nrestart", "Goal", "Another\nset piece", "Retention", "Second\nball", "Goal\nagainst"]
    keys = ["reach_danger", "peak_xshot", "xg", "goal", "set_piece", "retention", "second_ball", "goal_against"]
    fmt = ["{:.0%}", "{:.0%}", "{:.3f}", "{:.1%}", "{:.0%}", "{:.0%}", "{:.0%}", "{:.1%}"]
    rows, ref = [], []
    for rt in list(TCOL):
        g, lg = c.A[c.A.restart == rt], c.d[c.d.restart == rt]
        if len(g) < 3:
            continue
        rows.append([f"{rt} ({len(g)})"] + [g[k].mean() for k in keys])
        ref.append([lg[k].mean() for k in keys])
    p = table_page(c, f"{c.team}: what each restart type produces", "Per restart, 30 s live window  ·  coral = better than the "
                   "league average for that type, blue = worse", rows, [x.replace("\n", " ") for x in cols], fmt,
                   vs_colour(ref, invert=(7,)))
    return p


@viz
def type_profile_table_def(c):
    keys = ["reach_danger", "peak_xshot", "xg", "goal", "set_piece", "retention", "second_ball"]
    cols = ["Reach danger", "Peak xShot", "xG per restart", "Goal", "Another set piece", "Opp. retention", "Opp. second ball"]
    fmt = ["{:.0%}", "{:.0%}", "{:.3f}", "{:.1%}", "{:.0%}", "{:.0%}", "{:.0%}"]
    rows, ref = [], []
    for rt in list(TCOL):
        g, lg = c.D[c.D.restart == rt], c.d[c.d.restart == rt]
        if len(g) < 3:
            continue
        rows.append([f"{rt} ({len(g)})"] + [g[k].mean() for k in keys])
        ref.append([lg[k].mean() for k in keys])
    return table_page(c, f"{c.team}: what opponents' restarts produce against them", "Opponent restarts in their matches, per "
                      "restart  ·  coral = they concede less than the league average, blue = more", rows, cols, fmt,
                      vs_colour(ref, invert=tuple(range(7))))


# ---------------------------------------------------------------- league comparison
@viz
def league_attack_defence(c):
    return league_scatter(c, "sp_xg_pm", "sp_xg_against_pm", "Set-piece xG per match", "Set-piece xG conceded per match",
                          f"{c.team} against the league: set-piece attack vs defence", "Conceded axis inverted: top right is best",
                          invert_y=True)


@viz
def league_xg_goals(c):
    t = c.tab.assign(xg_tot=c.tab.sp_xg_pm * c.tab.matches)
    c.tab["xg_tot"] = t.xg_tot
    return league_scatter(c, "xg_tot", "sp_goals", "Set-piece xG (season)", "Set-piece goals", f"{c.team}: set-piece goals vs xG",
                          "Above the diagonal = scored more than the chances suggest", diag=True)


@viz
def league_corner_volume_quality(c):
    return league_scatter(c, "corners_pm", "corner_xg", "Corners per match", "xG per corner", f"{c.team}: corner volume vs quality",
                          "Corners and short corners, 30 s window")


@viz
def league_corner_targeting(c):
    return league_scatter(c, "corner_goal_area", "corner_xg", "Share of corners into the goal area", "xG per corner",
                          f"{c.team}: corner targeting", "Goal area: near post, six-yard centre, far post, penalty spot", pct=True)


@viz
def league_first_touch(c):
    return league_scatter(c, "first_touch", "first_touch_def", "First touch won on own corners", "First touch won on opponent corners",
                          f"{c.team}: who wins the first touch at corners", "First action after the delivery belongs to the team "
                          "(attacking) or to them when defending", pct=True)


@viz
def league_second_ball(c):
    return league_scatter(c, "second_ball", "retention", "Second balls won", "Retention after the restart",
                          f"{c.team}: control after set pieces", "Attacking restarts", pct=True)


@viz
def rank_xg(c):
    return league_bar(c, "sp_xg_pm", f"{c.team}: set-piece xG per match in the league", "Attacking restarts and penalties")


@viz
def rank_xg_against(c):
    return league_bar(c, "sp_xg_against_pm", f"{c.team}: set-piece xG conceded per match", "Lowest at the top", low_good=True)


@viz
def rank_corner_xg(c):
    return league_bar(c, "corner_xg", f"{c.team}: xG per corner in the league", "Corners and short corners", fmt="{:.3f}")


@viz
def rank_goal_diff(c):
    return league_bar(c, "sp_goal_diff", f"{c.team}: set-piece goal difference", "Scored minus conceded within 30 s of attacking "
                      "restarts", fmt="{:+.0f}")


PIZ = [("sp_xg_pm", "xG/match", False), ("sp_shots_pm", "Shots/match", False), ("corner_xg", "xG/corner", False),
       ("reach_danger", "Reach danger", False), ("peak_xshot", "Peak xShot", False), ("fk_xg", "xG/FK", False),
       ("throw_xg", "xG/throw", False), ("second_ball", "Second balls", False), ("retention", "Retention", False),
       ("first_touch", "1st touch att.", False), ("sp_xg_against_pm", "xG against", True), ("corner_xg_against", "xG/corner ag.", True),
       ("danger_against", "Danger ag.", True), ("first_touch_def", "1st touch def.", False)]


@viz
def pizza(c):
    t = c.tab
    pct = [((t[k] > t.at[c.team, k]).mean() if inv else (t[k] < t.at[c.team, k]).mean()) * 100 for k, _, inv in PIZ]
    p = page(c, f"{c.team}: the set-piece profile", f"Percentile within the league ({len(t)} clubs)  ·  defending metrics "
             "inverted: high = concedes little", h=9.0)
    ax = p.axes(top=1.9, bottom=1.0, left=2.0, right=2.0, projection="polar")
    ax.set_facecolor(BG)
    th = np.linspace(0, 2 * np.pi, len(PIZ), endpoint=False)
    wd = 2 * np.pi / len(PIZ) * 0.92
    cols = [ACCENT] * 7 + [GOLD] * 3 + [TEAL] * 4
    ax.bar(th, [100] * len(th), width=wd, color=CARD, edgecolor=BG)
    ax.bar(th, pct, width=wd, color=cols, edgecolor=BG)
    for t_, v, (_, lab, _) in zip(th, pct, PIZ):
        ax.text(t_, 117, lab, ha="center", va="center", fontsize=7.8, color=INK)
        ax.text(t_, max(v - 9, 10), f"{v:.0f}", ha="center", va="center", fontsize=7.5, color=BG, weight="bold")
    ax.set_ylim(0, 125), ax.set_yticks([]), ax.set_xticks([]), ax.spines["polar"].set_visible(False)
    ax.set_theta_offset(np.pi / 2), ax.set_theta_direction(-1)
    p.legend([("Threat", ACCENT), ("Control", GOLD), ("Defending", TEAL)], top=7.9, left=3.0)
    return p


# ---------------------------------------------------------------- corners
def corner_map(c, d, title, sub):
    p = page(c, title, sub, h=7.6)
    pt, ax = half(p)
    d = d[d.end_x.notna()]
    sc = pt.scatter(px(d.end_x), py(d.fold_y), s=30 + 900 * d.xg_for.clip(upper=0.5), c=d.peak_xd_for.clip(upper=0.4),
                    cmap=DANGER_CMAP, vmin=0, vmax=0.4, edgecolor=np.where(d.goals_for > 0, INK, HAIR),
                    lw=np.where(d.goals_for > 0, 1.6, 0.4), ax=ax, zorder=3)
    pt.scatter([105], [66], marker="^", s=80, color=MUTED2, ax=ax)
    legend(ax, [dot(ACCENT, "High peak danger"), dot(GOLD, "Some danger"), dot(GREY, "None"), dot(INK, "White ring = goal", hollow=True)])
    return p


@viz
def corners_attack_map(c):
    d = c.A[c.A.restart.isin(["Corner", "Short corner"])]
    return corner_map(c, d, f"{c.team}: where their corners land", f"{len(d)} corners, all folded to the top-right flag (triangle)  ·  "
                      "colour = peak xDanger in the next 30 s, size = xG")


@viz
def corners_defend_map(c):
    d = c.D[c.D.restart.isin(["Corner", "Short corner"])]
    return corner_map(c, d, f"{c.team}: where opponents' corners land", f"{len(d)} corners against, opponent's view, folded  ·  "
                      "colour = peak danger, size = xG conceded")


def zone_table(c, d, lg, title, sub, invert=False):
    rows, ref = [], []
    for z in CORNER_ZONES:
        g, l_ = d[d.zone == z], lg[lg.zone == z]
        rows.append([z.capitalize(), len(g) / max(len(d), 1), g.xg_for.mean() if len(g) else np.nan, g.reach_danger.mean() if len(g) else np.nan,
                     g.goals_for.sum()])
        ref.append([len(l_) / max(len(lg), 1), l_.xg_for.mean(), l_.reach_danger.mean(), None])
    inv = (0, 1, 2) if invert else ()
    return table_page(c, title, sub, rows, ["Share", "xG per corner", "Reach danger", "Goals"], ["{:.0%}", "{:.3f}", "{:.0%}", "{:.0f}"],
                      vs_colour(ref, invert=inv))


@viz
def corner_zones(c):
    lg = c.d[c.d.restart == "Corner"]
    return zone_table(c, c.A[c.A.restart == "Corner"], lg, f"{c.team}: corner target zones", "Delivered corners by target zone "
                      "vs the league  ·  coral = above league average")


@viz
def corner_zones_def(c):
    lg = c.d[c.d.restart == "Corner"]
    return zone_table(c, c.D[c.D.restart == "Corner"], lg, f"{c.team}: where opponents aim corners at them", "Corners against by "
                      "target zone  ·  coral = they concede less than average there", invert=True)


def option_table(c, d, lg, rt, title, sub, invert=False, min_n=2):
    keys = ["xg", "reach_danger", "retention", "second_ball", "goal"]
    rows, ref = [], []
    for op, g in d[d.restart == rt].groupby("option"):
        if len(g) < min_n:
            continue
        l_ = lg[(lg.restart == rt) & (lg.option == op)]
        rows.append([f"{op} ({len(g)})"] + [g[k].mean() for k in keys])
        ref.append([l_[k].mean() for k in keys])
    if not rows:
        raise ValueError(f"no {rt}")
    order = np.argsort([-r[1] for r in rows])
    rows, ref = [rows[i] for i in order], [ref[i] for i in order]
    return table_page(c, title, sub, rows, ["xG per restart", "Reach danger", "Retention", "Second ball", "Goal"],
                      ["{:.3f}", "{:.0%}", "{:.0%}", "{:.0%}", "{:.0%}"], vs_colour(ref, invert=(0, 1, 4) if invert else ()))


@viz
def corner_options(c):
    return option_table(c, c.A, c.d, "Corner", f"{c.team}: corner options", "Swing and target zone, per corner  ·  coral = better "
                        "than the league for the same option  ·  count in brackets")


@viz
def short_corner_options(c):
    return option_table(c, c.A, c.d, "Short corner", f"{c.team}: what happens after short corners", "Short corners by what follows")


@viz
def corner_options_def(c):
    return option_table(c, c.D, c.d, "Corner", f"{c.team}: corner options used against them", "Opponents' corners in their "
                        "matches  ·  coral = they concede less than the league for that option", invert=True)


@viz
def corner_side_swing(c):
    d = c.A[c.A.restart == "Corner"]
    g = d.groupby(["side", "swing"]).agg(n=("xg_for", "size"), xg=("xg_for", "mean"), goals=("goals_for", "sum"))
    lg = c.d[c.d.restart == "Corner"].groupby(["side", "swing"]).xg_for.mean()
    p = page(c, f"{c.team}: corners by side and swing", "Left / right flag (taker's view) and inswing / outswing: count, xG per "
             "corner, league xG per corner (outline)")
    ax = axes(p, left=2.4)
    y = np.arange(len(g))[::-1]
    ax.barh(y, g.xg, color=[ACCENT if s == "Inswing" else TEAL for _, s in g.index], height=0.6)
    ax.barh(y, lg.reindex(g.index).values, fill=False, edgecolor=INK, height=0.6)
    for yi, r in zip(y, g.itertuples()):
        ax.text(r.xg, yi, f"  {r.n} corners  ·  {int(r.goals)} goals", va="center", fontsize=8, color=INK)
    ax.set_yticks(y), ax.set_yticklabels([f"{s} flag, {w}" for s, w in g.index])
    ax.set_xlabel("xG per corner")
    clean(ax, grid="x")
    return p


def takers(c, d, title, sub):
    g = d.groupby("player_name").agg(n=("xg_for", "size"), xg=("xg_for", "sum"), dang=("reach_danger", "mean")).sort_values("n").tail(12)
    p = page(c, title, sub)
    ax = axes(p, left=2.4, right=1.6)
    y = np.arange(len(g))
    ax.barh(y, g.n, color=GOLD, height=0.65)
    for yi, r in zip(y, g.itertuples()):
        ax.text(r.n, yi, f"  {r.n}  ·  {r.xg:.2f} xG  ·  {r.dang:.0%} reach danger", va="center", fontsize=7.8, color=INK)
    ax.set_yticks(y), ax.set_yticklabels(g.index, fontsize=8.5)
    clean(ax, grid="x")
    return p


@viz
def corner_takers(c):
    return takers(c, c.A[c.A.restart.isin(["Corner", "Short corner"])], f"{c.team}: corner takers", "Corners taken, xG in the next 30 s "
                  "and share reaching danger")


@viz
def first_touch_map(c):
    f = c.first[(c.first.team == c.team) & c.first.n_x.notna()]
    p = page(c, f"{c.team}: who gets the first touch at their corners", f"First action after the delivery: {f.won.mean():.0%} theirs "
             f"(league {c.first.won.mean():.0%})  ·  location of that touch, folded", h=7.6)
    pt, ax = half(p)
    fy = np.where(f.y >= 0, 1, -1)
    own = f.won == 1
    # opponent touches are recorded from the opponent's side: mirror them into the attacking frame
    x = np.where(own, f.n_x, -f.n_x)
    y = np.where(own, f.n_y, -f.n_y) * fy
    pt.scatter(px(x[own]), py(y[own]), s=30, color=ACCENT, ax=ax, zorder=3, alpha=0.85)
    pt.scatter(px(x[~own]), py(y[~own]), s=30, color=BLUE, ax=ax, zorder=3, alpha=0.85)
    legend(ax, [dot(ACCENT, f"Their touch ({int(own.sum())})"), dot(BLUE, f"Opponent touch ({int((~own).sum())})")], ncol=2)
    return p


@viz
def first_touch_map_def(c):
    f = c.first[(c.first.opponent == c.team) & c.first.n_x.notna()]
    p = page(c, f"{c.team}: first touch when defending corners", f"Opponent corners: {1 - f.won.mean():.0%} first touches by "
             f"{c.team} (league {1 - c.first.won.mean():.0%})  ·  opponent's attacking view, folded", h=7.6)
    pt, ax = half(p)
    fy = np.where(f.y >= 0, 1, -1)
    own = f.won == 1
    x = np.where(own, f.n_x, -f.n_x)
    y = np.where(own, f.n_y, -f.n_y) * fy
    pt.scatter(px(x[~own]), py(y[~own]), s=30, color=TEAL, ax=ax, zorder=3, alpha=0.85)
    pt.scatter(px(x[own]), py(y[own]), s=30, color=ACCENT, ax=ax, zorder=3, alpha=0.85)
    legend(ax, [dot(TEAL, f"{c.team} touch ({int((~own).sum())})"), dot(ACCENT, f"Opponent touch ({int(own.sum())})")], ncol=2)
    return p


@viz
def corner_zone_first_touch(c):
    f = c.first[c.first.zone.notna()]
    me = f[f.team == c.team].groupby("zone").won.agg(["mean", "size"]).reindex(CORNER_ZONES)
    de = f[f.opponent == c.team].groupby("zone").won.agg(["mean", "size"]).reindex(CORNER_ZONES)
    lg = f.groupby("zone").won.mean().reindex(CORNER_ZONES)
    p = page(c, f"{c.team}: first touch by target zone", "Share of first touches won: attacking corners (coral), defending (teal, "
             "= 1 minus opponent share), league attacking average (outline)")
    ax = axes(p, left=2.2)
    y = np.arange(len(CORNER_ZONES))[::-1]
    ax.barh(y + 0.2, me["mean"], height=0.38, color=ACCENT)
    ax.barh(y - 0.2, 1 - de["mean"], height=0.38, color=TEAL)
    ax.barh(y + 0.2, lg, height=0.38, fill=False, edgecolor=INK)
    for yi, n1, n2 in zip(y, me["size"].fillna(0), de["size"].fillna(0)):
        ax.text(1.01, yi, f"n={int(n1)} / {int(n2)}", va="center", fontsize=7.5, color=MUTED2)
    ax.set_yticks(y), ax.set_yticklabels([z.capitalize() for z in CORNER_ZONES])
    ax.set_xlim(0, 1.15), ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    clean(ax, grid="x")
    return p


# ---------------------------------------------------------------- shots from set pieces
def shot_map(c, s, title, sub):
    p = page(c, title, sub, h=7.6)
    pt, ax = half(p, pad_bottom=-14)
    for rt, col in TCOL.items():
        e = s[s.restart == rt]
        if len(e):
            pt.scatter(px(e.x), py(e.y), s=30 + 900 * e.xg, color=col, edgecolor=np.where(e.goal == 1, INK, BG),
                       lw=np.where(e.goal == 1, 1.6, 0.5), ax=ax, zorder=3, alpha=0.9)
    legend(ax, [dot(col, f"{rt} ({int((s.restart == rt).sum())})") for rt, col in TCOL.items() if (s.restart == rt).any()]
           + [dot(INK, "White ring = goal", hollow=True)], ncol=3)
    return p


@viz
def shots_for_map(c):
    s = c.shA
    return shot_map(c, s, f"{c.team}: shots from set pieces", f"{len(s)} shots within 30 s of an attacking restart  ·  "
                    f"{s.xg.sum():.1f} xG, {int(s.goal.sum())} goals  ·  size = xG, colour = restart type")


@viz
def shots_against_map(c):
    s = c.shD
    return shot_map(c, s, f"{c.team}: shots conceded from set pieces", f"{len(s)} shots  ·  {s.xg.sum():.1f} xG, "
                    f"{int(s.goal.sum())} goals  ·  opponent's view")


def shooters(c, s, title, sub):
    g = s.groupby("player_name").agg(n=("xg", "size"), xg=("xg", "sum"), g=("goal", "sum")).sort_values("xg").tail(12)
    p = page(c, title, sub)
    ax = axes(p, left=2.4, right=1.4)
    y = np.arange(len(g))
    ax.barh(y, g.xg, color=ACCENT, height=0.65)
    for yi, r in zip(y, g.itertuples()):
        ax.text(r.xg, yi, f"  {r.n} shots  ·  {int(r.g)} goals", va="center", fontsize=7.8, color=INK)
    ax.set_yticks(y), ax.set_yticklabels(g.index, fontsize=8.5), ax.set_xlabel("xG")
    clean(ax, grid="x")
    return p


@viz
def set_piece_shooters(c):
    return shooters(c, c.shA, f"{c.team}: who shoots from set pieces", "xG from shots within 30 s of attacking restarts")


@viz
def set_piece_shooters_against(c):
    return shooters(c, c.shD, f"{c.team}: who hurt them from set pieces", "Opponent players: xG within 30 s of their attacking restarts")


@viz
def shot_timing(c):
    p = page(c, f"{c.team}: how fast set-piece shots come", "Seconds from the restart to the shot: first contact vs second phase")
    ax = axes(p)
    b = np.arange(0, 31, 2)
    ax.hist(c.shots.delay, bins=b, density=True, color=GREY, alpha=0.7, label="League")
    ax.hist(c.shA.delay, bins=b, density=True, histtype="step", color=ACCENT, lw=2, label="For")
    ax.hist(c.shD.delay, bins=b, density=True, histtype="step", color=BLUE, lw=2, label="Against")
    ax.set_xlabel("Seconds after the restart"), ax.legend(frameon=False, labelcolor=MUTED2, fontsize=8)
    clean(ax, grid="y")
    return p


# ---------------------------------------------------------------- free kicks and throw-ins
def delivery_map(c, d, title, sub, colour_col="xg_for"):
    p = page(c, title, sub, h=7.6)
    pt, ax = full(p)
    d = d[d.end_x.notna()]
    norm = Normalize(0, max(d[colour_col].quantile(0.95), 0.02)) if len(d) else Normalize(0, 1)
    pt.lines(px(d.x), py(d.y), px(d.end_x), py(d.end_y), ax=ax, color=GREY, lw=0.8, alpha=0.5, comet=True, zorder=2)
    pt.scatter(px(d.end_x), py(d.end_y), s=25, c=d[colour_col], cmap=DANGER_CMAP, norm=norm, edgecolor=np.where(d.goals_for > 0, INK, HAIR),
               lw=np.where(d.goals_for > 0, 1.6, 0.3), ax=ax, zorder=3)
    return p


@viz
def fk_attack_map(c):
    d = c.A[(c.A.restart == "Indirect free kick") & c.A.option.str.startswith("Opp. half")]
    return delivery_map(c, d, f"{c.team}: free kicks in the opponent half", f"{len(d)} free kicks: line from taker to end point, "
                        "end coloured by xG in the next 30 s")


@viz
def fk_defend_map(c):
    d = c.D[(c.D.restart == "Indirect free kick") & c.D.option.str.startswith("Opp. half")]
    return delivery_map(c, d, f"{c.team}: free kicks conceded in their half", f"{len(d)} opponent free kicks  ·  opponent's view")


@viz
def fk_options(c):
    return option_table(c, c.A, c.d, "Indirect free kick", f"{c.team}: free-kick options", "All indirect free kicks by option  ·  "
                        "coral = better than the league for the same option")


@viz
def dfk_map(c):
    d = pd.concat([c.A[c.A.restart == "Direct free kick"].assign(s="For"), c.D[c.D.restart == "Direct free kick"].assign(s="Against")])
    if d.empty:
        raise ValueError("no direct free kicks")
    p = page(c, f"{c.team}: direct free kicks", f"For (coral, {int((d.s == 'For').sum())}) and against (blue, {int((d.s == 'Against').sum())}), "
             "from the taker's position  ·  size = xG of the shot, white ring = goal", h=7.6)
    pt, ax = half(p, pad_bottom=-10)
    for s_, col in (("For", ACCENT), ("Against", BLUE)):
        e = d[d.s == s_]
        pt.scatter(px(e.x), py(e.y), s=40 + 900 * e.own_xg.fillna(0), color=col, edgecolor=np.where(e.own_goal == 1, INK, BG),
                   lw=np.where(e.own_goal == 1, 1.6, 0.5), ax=ax, zorder=3, alpha=0.9)
    return p


@viz
def fk_takers(c):
    return takers(c, c.A[c.A.restart.isin(["Indirect free kick", "Direct free kick"]) & c.A.attacking], f"{c.team}: free-kick takers",
                  "Attacking free kicks taken")


@viz
def throw_map(c):
    d = c.A[(c.A.restart == "Throw-in") & c.A.option.str.startswith("Final third")]
    return delivery_map(c, d, f"{c.team}: final-third throw-ins", f"{len(d)} throw-ins: line from thrower to end point, "
                        "end coloured by peak xShot in 30 s", colour_col="peak_xshot")


@viz
def throw_options(c):
    return option_table(c, c.A, c.d, "Throw-in", f"{c.team}: throw-in options", "All throw-ins by option  ·  coral = better than "
                        "the league for the same option")


@viz
def throw_retention_map(c):
    d = c.A[c.A.restart == "Throw-in"]
    p = page(c, f"{c.team}: keeping the ball from throw-ins", f"{len(d)} throw-ins by touchline position: retention after 30 s "
             "(colour) and count (label)", h=7.6)
    pt, ax = full(p)
    bs = pt.bin_statistic(px(d.x), py(d.y.abs() * np.sign(d.y)), values=d.retention, statistic="mean", bins=(6, 2))
    n = pt.bin_statistic(px(d.x), py(d.y), statistic="count", bins=(6, 2))["statistic"]
    bs["statistic"] = np.where(n >= 3, bs["statistic"], np.nan)
    pt.heatmap(bs, ax=ax, cmap=HEAT, vmin=0, vmax=1, edgecolor=BG, lw=1.5, zorder=1)
    for cx, cy, v, k in zip(bs["cx"].ravel(), bs["cy"].ravel(), bs["statistic"].ravel(), n.ravel()):
        if np.isfinite(v):
            pt.annotate(f"{v:.0%}\nn={int(k)}", xy=(cx, cy), ax=ax, ha="center", va="center", fontsize=8, color=INK, zorder=3)
    return p


@viz
def throw_takers(c):
    return takers(c, c.A[c.A.restart == "Throw-in"], f"{c.team}: throw-in takers", "All throw-ins")


# ---------------------------------------------------------------- goal kicks, kick-offs, penalties
@viz
def goal_kick_map(c):
    d = c.A[c.A.restart == "Goal kick"]
    p = page(c, f"{c.team}: goal kicks", f"{len(d)} goal kicks: end points coloured by retention after 30 s (coral = kept), "
             f"{d.retention.mean():.0%} kept (league {c.d[c.d.restart == 'Goal kick'].retention.mean():.0%})", h=7.6)
    pt, ax = full(p)
    d = d[d.end_x.notna()]
    pt.lines(px(d.x), py(d.y), px(d.end_x), py(d.end_y), ax=ax, color=GREY, lw=0.6, alpha=0.35, comet=True, zorder=2)
    pt.scatter(px(d.end_x), py(d.end_y), s=22, color=np.where(d.retention >= 0.5, ACCENT, BLUE), ax=ax, zorder=3)
    legend(ax, [dot(ACCENT, "Kept"), dot(BLUE, "Lost")], ncol=2)
    return p


@viz
def goal_kick_options(c):
    return option_table(c, c.A, c.d, "Goal kick", f"{c.team}: goal-kick options", "Goal kicks by option")


@viz
def kick_off_options(c):
    return option_table(c, c.A, c.d, "Kick-off", f"{c.team}: kick-off options", "Kick-offs by option")


@viz
def penalties(c):
    d = pd.concat([c.A[c.A.restart == "Penalty"].assign(s="For"), c.D[c.D.restart == "Penalty"].assign(s="Against")])
    if d.empty:
        raise ValueError("no penalties")
    p = page(c, f"{c.team}: penalties", f"For: {int(d[d.s == 'For'].own_goal.sum())} of {int((d.s == 'For').sum())} scored  ·  "
             f"against: {int(d[d.s == 'Against'].own_goal.sum())} of {int((d.s == 'Against').sum())}  ·  seen from the taker", h=6.2)
    ax = axes(p, top=1.95, bottom=1.2, left=1.4, right=1.4)
    ax.add_patch(Rectangle((-3.66, 0), 7.32, 2.44, fill=False, ec=INK, lw=2))
    ax.plot([-5, 5], [0, 0], color=HAIR)
    for s_, col in (("For", ACCENT), ("Against", BLUE)):
        e = d[d.s == s_]
        ax.scatter(-e.target_y, e.target_z.clip(lower=0.06), s=160, color=np.where(e.own_goal == 1, col, "none"), edgecolor=col, lw=1.5)
    ax.set_xlim(-5, 5), ax.set_ylim(-0.2, 3.2), ax.set_aspect("equal"), ax.axis("off")
    legend(ax, [dot(ACCENT, "For: scored"), dot(ACCENT, "For: missed", hollow=True), dot(BLUE, "Against: scored"),
                dot(BLUE, "Against: missed", hollow=True)], y=0.0)
    return p


# ---------------------------------------------------------------- context splits
@viz
def by_minute(c):
    a = c.A[c.A.attacking].assign(b=(c.A.minute.clip(upper=94) // 15).astype(int)).groupby("b").xg_for.sum().reindex(range(7), fill_value=0)
    o = c.D[c.D.attacking].assign(b=(c.D.minute.clip(upper=94) // 15).astype(int)).groupby("b").xg_for.sum().reindex(range(7), fill_value=0)
    p = page(c, f"{c.team}: set-piece xG by match period", "Per 15-minute block: created (coral) and conceded (blue)")
    ax = axes(p)
    x = np.arange(7)
    ax.bar(x - 0.2, a, width=0.38, color=ACCENT), ax.bar(x + 0.2, o, width=0.38, color=BLUE)
    ax.set_xticks(x), ax.set_xticklabels(["0-15", "15-30", "30-45+", "45-60", "60-75", "75-90", "90+"])
    clean(ax, grid="y")
    return p


@viz
def home_away(c):
    rows = []
    for ha, flag in (("Home", True), ("Away", False)):
        mids = [m for m in c.matches if (c.home.get(m) == c.team) == flag]
        a = c.A[c.A.attacking & c.A.match_id.isin(mids)]
        o = c.D[c.D.attacking & c.D.match_id.isin(mids)]
        n = max(len(mids), 1)
        rows.append((ha, len(mids), a.xg_for.sum() / n, o.xg_for.sum() / n, a.goals_for.sum(), o.goals_for.sum(),
                     c.A[c.A.match_id.isin(mids) & c.A.restart.isin(["Corner", "Short corner"])].shape[0] / n))
    t = pd.DataFrame(rows, columns=["", "matches", "xg", "xga", "g", "ga", "corners"]).set_index("")
    p = page(c, f"{c.team}: set pieces home and away", "Per match: xG for and against, corners; season goals in labels")
    ax = axes(p, left=1.4)
    x = np.arange(2)
    ax.bar(x - 0.27, t.xg, width=0.26, color=ACCENT, label="xG for per match")
    ax.bar(x, t.xga, width=0.26, color=BLUE, label="xG against per match")
    ax2 = ax.twinx()
    ax2.bar(x + 0.27, t.corners, width=0.26, color=GOLD, label="Corners per match")
    ax2.tick_params(colors=GOLD, length=0)
    for sp in ax2.spines.values():
        sp.set_visible(False)
    ax.set_xticks(x), ax.set_xticklabels([f"{i} ({int(r.matches)} matches)\n{int(r.g)} scored, {int(r.ga)} conceded" for i, r in t.iterrows()])
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, labelcolor=MUTED2, fontsize=8, loc="upper right")
    clean(ax, grid="y")
    return p


@viz
def opponents_table(c):
    m = by_match(c)
    rows = [[f"{pd.to_datetime(r.date):%d %b} {r.ha} {r.opp}", r.xg, r.xg_ag, r.g, r.g_ag] for r in m.itertuples()]
    ref = [[m.xg.mean(), m.xg_ag.mean(), None, None] for _ in rows]
    return table_page(c, f"{c.team}: set pieces, opponent by opponent", "xG and goals within 30 s of attacking restarts  ·  coral "
                      "= better than their own average", rows, ["xG for", "xG against", "Goals for", "Goals against"],
                      ["{:.2f}", "{:.2f}", "{:.0f}", "{:.0f}"], vs_colour(ref, invert=(1,)))


@viz
def sp_share(c):
    s = pd.read_parquet(c.ws.scored, columns=["match_id", "squad_id", "action_type", "xg"])
    s = s[s.action_type == "SHOT"]
    sm = c.d.drop_duplicates("squad_id").set_index("team").squad_id
    tot = s.groupby("squad_id").xg.sum()
    tm = c.d.drop_duplicates("team").set_index("team")
    share_for = c.shots.groupby("r_team").xg.sum() / tot.rename(index={v: k for k, v in sm.items()})
    conc = {}
    for team in c.tab.index:
        mids = c.d[(c.d.team == team) | (c.d.opponent == team)].match_id.unique()
        conc[team] = s[s.match_id.isin(mids) & (s.squad_id != sm[team])].xg.sum()
    share_ag = c.shots.groupby("r_opponent").xg.sum() / pd.Series(conc)
    t = pd.DataFrame({"for": share_for, "against": share_ag}).dropna().sort_values("for")
    p = page(c, f"{c.team}: how much of their game is set pieces", "Share of all xG that came within 30 s of attacking restarts: "
             "created (coral) and conceded (blue)", h=2.4 + 0.27 * len(t))
    ax = axes(p, top=1.85, left=2.6, right=0.8)
    y = np.arange(len(t))
    ax.barh(y + 0.18, t["for"], height=0.34, color=[ACCENT if i == c.team else GREY for i in t.index])
    ax.barh(y - 0.18, t["against"], height=0.34, color=[BLUE if i == c.team else "#2E3B48" for i in t.index])
    ax.set_yticks(y), ax.set_yticklabels(t.index, fontsize=8)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    clean(ax, grid="x")
    return p


@viz
def sequences(c):
    d = c.A[c.A.attacking & c.A.signature.notna()]
    d = d.assign(seq=d.signature.str.split(" > ").str[:4].str.join(" > "))
    g = d.groupby("seq").agg(n=("xg_for", "size"), xg=("xg_for", "mean"), dang=("reach_danger", "mean")).query("n >= 3")
    g = g.sort_values("n").tail(14)
    if g.empty:
        raise ValueError("no sequences")
    p = page(c, f"{c.team}: what their set pieces turn into", "Most common first steps of the possession after attacking restarts "
             "(3+ times): count, xG per restart, share reaching danger", h=2.6 + 0.36 * len(g))
    ax = axes(p, top=1.9, left=4.6, right=1.6)
    y = np.arange(len(g))
    ax.barh(y, g.n, color=GOLD, height=0.65)
    for yi, r in zip(y, g.itertuples()):
        ax.text(r.n, yi, f"  {r.n}  ·  {r.xg:.3f} xG  ·  {r.dang:.0%}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y), ax.set_yticklabels(g.index, fontsize=7.3)
    clean(ax, grid="x")
    return p


@viz
def setup_time(c):
    d = c.A[c.A.attacking]
    lg = c.d[c.d.attacking]
    p = page(c, f"{c.team}: how quickly they take set pieces", "Seconds from the restart being awarded to being taken "
             "(setup time), per restart type: theirs (coral) vs league (grey)")
    ax = axes(p, left=2.0)
    types = [t for t in TYPES if (d.restart == t).sum() >= 3]
    y = np.arange(len(types))[::-1]
    ax.barh(y + 0.2, [d[d.restart == t].setup_s.median() for t in types], height=0.38, color=ACCENT)
    ax.barh(y - 0.2, [lg[lg.restart == t].setup_s.median() for t in types], height=0.38, color=GREY)
    ax.set_yticks(y), ax.set_yticklabels(types), ax.set_xlabel("Median setup time (s)")
    clean(ax, grid="x")
    return p


@viz
def peak_xshot_dist(c):
    p = page(c, f"{c.team}: how close set pieces get to a shot", "Peak xShot within 30 s of attacking restarts: theirs vs "
             "league vs conceded")
    ax = axes(p)
    b = np.linspace(0, 1, 21)
    ax.hist(c.d[c.d.attacking].peak_xshot.dropna(), bins=b, density=True, color=GREY, alpha=0.7, label="League")
    ax.hist(c.A[c.A.attacking].peak_xshot.dropna(), bins=b, density=True, histtype="step", color=ACCENT, lw=2, label="For")
    ax.hist(c.D[c.D.attacking].peak_xshot.dropna(), bins=b, density=True, histtype="step", color=BLUE, lw=2, label="Against")
    ax.set_xlabel("Peak xShot"), ax.legend(frameon=False, labelcolor=MUTED2, fontsize=8)
    clean(ax, grid="y")
    return p


@viz
def pillars(c):
    keys = [("second_ball", "Second ball"), ("retention", "Retention"), ("dominance", "Dominance"), ("set_piece", "Another set piece"),
            ("reach_danger", "Reach danger")]
    a = c.A[c.A.attacking]
    o = c.D[c.D.attacking]
    lg = c.d[c.d.attacking]
    p = page(c, f"{c.team}: the restart pillars", "Attacking restarts: theirs (coral), opponents' against them (blue), league (grey)")
    ax = axes(p, left=2.0)
    y = np.arange(len(keys))[::-1]
    ax.barh(y + 0.27, [a[k].mean() for k, _ in keys], height=0.26, color=ACCENT)
    ax.barh(y, [o[k].mean() for k, _ in keys], height=0.26, color=BLUE)
    ax.barh(y - 0.27, [lg[k].mean() for k, _ in keys], height=0.26, color=GREY)
    ax.set_yticks(y), ax.set_yticklabels([lab for _, lab in keys])
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    clean(ax, grid="x")
    return p


@viz
def all_restart_locations(c):
    d = c.A[c.A.x.notna()]
    p = page(c, f"{c.team}: where they take restarts", f"{len(d)} restarts by type at the taker's position", h=7.6)
    pt, ax = full(p)
    for rt, col in TCOL.items():
        e = d[d.restart == rt]
        pt.scatter(px(e.x), py(e.y), s=14, color=col, ax=ax, zorder=3, alpha=0.8)
    legend(ax, [dot(col, f"{rt} ({int((d.restart == rt).sum())})") for rt, col in TCOL.items()], ncol=4)
    return p


@viz
def danger_heat(c):
    d = c.A[c.A.x.notna()]
    p = page(c, f"{c.team}: where their restarts become dangerous", "Mean peak xDanger in 30 s by restart location "
             "(zones with 3+ restarts)", h=7.6)
    pt, ax = full(p)
    bs = pt.bin_statistic(px(d.x), py(d.y), values=d.peak_xd_for, statistic="mean", bins=(8, 6))
    n = pt.bin_statistic(px(d.x), py(d.y), statistic="count", bins=(8, 6))["statistic"]
    bs["statistic"] = np.where(n >= 3, bs["statistic"], np.nan)
    pt.heatmap(bs, ax=ax, cmap=HEAT, edgecolor=BG, lw=1.5, zorder=1)
    for cx, cy, v in zip(bs["cx"].ravel(), bs["cy"].ravel(), bs["statistic"].ravel()):
        if np.isfinite(v):
            pt.annotate(f"{v:.3f}", xy=(cx, cy), ax=ax, ha="center", va="center", fontsize=7.5, color=INK, zorder=3)
    return p


# ---------------------------------------------------------------- run
def run(league, team, out_root=None):
    c = Ctx(league, team)
    folder = Path(out_root or config.OUT_ROOT) / league / f"{c.team} set pieces"
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("[0-9][0-9]_*.png"):
        old.unlink()
    done, failed, n = [], [], 0
    for fn in VIS:
        try:
            p = fn(c)
            n += 1
            p.save(folder / f"{n:02d}_{fn.__name__}.png")
            done.append(fn.__name__)
        except Exception as e:  # noqa: BLE001
            failed.append((fn.__name__, f"{type(e).__name__}: {e}"))
        plt.close("all")
    keep = ["date", "match_id", "idx", "team", "opponent", "home", "minute", "player_name", "restart", "option", "x", "y", "end_x", "end_y",
            "xg_for", "goals_for", "shots_for", "peak_xd_for", "peak_xshot", "reach_danger", "retention", "second_ball", "dominance",
            "set_piece", "xg_against", "goals_against", "signature"]
    c.A[keep].to_csv(folder / "restarts_for.csv", index=False)
    c.D[keep].to_csv(folder / "restarts_against.csv", index=False)
    c.tab.round(4).to_csv(folder / "league_set_piece_table.csv")
    print(f"{len(done)} visuals saved to {folder}")
    for name, err in failed:
        print(f"  skipped {name}: {err}")
    return folder
