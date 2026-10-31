"""Post-match report for one match: short form (1 page, portrait) and long form (landscape pages).

Built from the package's models: xG and xGOT per shot, xDanger per action (danger created, momentum, creators),
restart windows (set pieces and restarts in the match), and league context from every team-match of the season.
"""
import numpy as np
import pandas as pd

from .. import style
from . import draw as D
from .base import Report, slug
from .data import TEAM_METRICS, league, pct_rank

STAT_ROWS = ["goals", "xg", "shots", "on_target", "xgot", "xdanger", "possession", "passes", "pass_pct", "progressive",
             "final_third", "box_entries", "crosses", "field_tilt", "regains", "high_regains", "def_actions", "ppda",
             "sp_xg", "corners", "fouls"]


class Match:
    def __init__(self, comp, match=None, team=None):
        self.L = L = league(comp)
        self.comp = comp
        self.m = m = L.match(match, team)
        self.mid, self.home, self.away = m.match_id, m.home, m.away
        a = L.actions[L.actions.match_id == self.mid].sort_values("idx")
        self.a = a
        self.shots = L.shots[L.shots.match_id == self.mid].sort_values(["period", "t"])
        self.restarts = L.restarts[L.restarts.match_id == self.mid]
        tm = L.team_match.set_index(["match_id", "team"])
        self.hs, self.as_ = tm.loc[(self.mid, self.home)], tm.loc[(self.mid, self.away)]
        self.hg, self.ag = int(m.home_goals), int(m.away_goals)
        self.md = D.matchday(m.matchday)
        self.date = pd.to_datetime(m.date)
        self.goals = self._goals()

    # ------------------------------------------------------------------ text
    def headline(self):
        h, a, hg, ag = self.home, self.away, self.hg, self.ag
        if hg == ag:
            return f"{h} and {a} draw {hg}-{ag}"
        return f"{h} beat {a} {hg}-{ag}" if hg > ag else f"{a} win {ag}-{hg} away at {h}"

    def subline(self):
        poss = self.hs.possession
        return (f"xG {self.hs.xg:.2f} - {self.as_.xg:.2f}  ·  Shots {self.hs.shots} - {self.as_.shots}  ·  "
                f"Possession {poss:.0%} - {1 - poss:.0%}  ·  {self.date:%d %B %Y}")

    def eyebrow(self, kind):
        return "  ·  ".join([x for x in (self.comp, self.md, kind) if x])

    def _goals(self):
        a = self.a
        g = a[((a.action_type == "SHOT") & (a.result == "SUCCESS")) | (a.action_type == "OWN_GOAL")].copy()
        g["scoring_team"] = np.where(g.action_type == "OWN_GOAL", np.where(g.team == self.home, self.away, self.home), g.team)
        g["label"] = [f"{D.short_name(p)} {int(np.ceil(mn))}'" + (" (og)" if t == "OWN_GOAL" else " (pen)" if act == "PENALTY_KICK" else "")
                      for p, mn, t, act in zip(g.player_name, g.minute, g.action_type, g.action)]
        return g

    def side(self, team):
        return self.a[self.a.team == team]


def _short_name(n):
    if not isinstance(n, str):
        return "?"
    parts = n.split()
    return parts[-1] if len(parts) > 1 else n


D.short_name = _short_name


# ------------------------------------------------------------------ panels
def scoreboard(p, M, top, height):
    ax = p.axes(top=top, bottom=p.h - top - height, left=0.55, right=0.55)
    D.axis_off(ax)
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.02", fc=style.CARD, ec=style.HAIR, lw=0.8,
                                transform=ax.transAxes))
    D.text(ax, 0.40, 0.72, M.home, size=15, bold=True, serif=True, color=D.home_col(), ha="right")
    D.text(ax, 0.60, 0.72, M.away, size=15, bold=True, serif=True, color=D.away_col(), ha="left")
    D.text(ax, 0.5, 0.68, f"{M.hg}  -  {M.ag}", size=26, bold=True, serif=True, ha="center")
    ht = f"HT {int(M.m.home_ht)}-{int(M.m.away_ht)}" if pd.notna(M.m.home_ht) else ""
    D.text(ax, 0.5, 0.36, ht, size=8, color=style.MUTED, ha="center")
    for team, x, ha, s in ((M.home, 0.40, "right", M.hs), (M.away, 0.60, "left", M.as_)):
        form = M.m.home_formation if team == M.home else M.m.away_formation
        D.text(ax, x, 0.50, "  ·  ".join([f for f in (form, f"xG {s.xg:.2f}", f"xDanger {s.xdanger:.2f}") if f]), size=8,
               color=style.MUTED2, ha=ha)
        gl = M.goals[M.goals.scoring_team == team].label.tolist()
        for k, lab in enumerate(gl[:5]):
            D.text(ax, x, 0.34 - 0.11 * k, lab, size=7.8, ha=ha)


def xg_race(ax, M, legend=True):
    for team, col in ((M.home, D.home_col()), (M.away, D.away_col())):
        s = M.shots[M.shots.team == team]
        mins = np.r_[0, s.minute.values, max(95, M.a.minute.max())]
        cum = np.r_[0, s.xg.cumsum().values, s.xg.sum()]
        ax.step(mins, cum, where="post", color=col, lw=2, label=f"{team}  {s.xg.sum():.2f}")
        ax.fill_between(mins, cum, step="post", color=col, alpha=0.08)
        g = s[s.result == "SUCCESS"]
        for mn, x in zip(g.minute, g.xg):
            c = cum[np.searchsorted(mins, mn, side="right") - 1]
            ax.scatter(mn, c, s=55, color=col, edgecolor=style.INK, lw=1.2, zorder=4)
    for team, col in ((M.home, D.home_col()), (M.away, D.away_col())):
        og = M.goals[(M.goals.scoring_team == team)]
        for mn, lab in zip(og.minute, og.label):
            s = M.shots[(M.shots.team == team) & (M.shots.minute <= mn + 1e-6)]
            ax.annotate(lab, (mn, s.xg.sum()), xytext=(4, 8), textcoords="offset points", fontsize=7.2, color=style.INK)
    ax.axvline(45, color=style.HAIR, lw=0.8, ls=(0, (3, 3)))
    ax.set_xlim(0, max(95, M.a.minute.max()) + 6), ax.set_ylim(0, None)  # room for late goal labels
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90]), ax.set_ylabel("Cumulative xG")
    D.clean(ax, grid="y")
    if legend:
        ax.legend(frameon=False, labelcolor=style.MUTED2, fontsize=8, loc="upper left")


def momentum(ax, M, step=5):
    a = M.a
    pos = np.where(a.in_poss == 1, a.xd_added.clip(lower=0), 0.0)
    b = (a.minute // step).astype(int) * step
    v = pd.DataFrame({"b": b, "team": a.team, "v": pos}).groupby(["b", "team"]).v.sum().unstack(fill_value=0)
    for t in (M.home, M.away):
        if t not in v:
            v[t] = 0.0
    x = v.index.values + step / 2
    ax.bar(x, v[M.home], width=step * 0.86, color=D.home_col())
    ax.bar(x, -v[M.away], width=step * 0.86, color=D.away_col())
    lim = max(v.values.max(), 0.05) * 1.25
    for _, g in M.goals.iterrows():
        up = g.scoring_team == M.home
        ax.scatter(g.minute, lim * 0.9 * (1 if up else -1), s=34, color=style.INK, marker="o", zorder=4)
    ax.axhline(0, color=style.HAIR, lw=0.8)
    ax.axvline(45, color=style.HAIR, lw=0.8, ls=(0, (3, 3)))
    ax.set_ylim(-lim, lim), ax.set_xlim(0, max(95, M.a.minute.max()))
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90]), ax.set_yticks([])
    D.clean(ax, grid=None)


def shot_map_full(ax, M, label_goals=True, scale=900):
    pt = D.pitch(ax)
    for team, flip in ((M.home, False), (M.away, True)):
        s = M.shots[M.shots.team == team]
        x, y = (-s.x, -s.y) if flip else (s.x, s.y)
        col = D.side_col(team == M.home)
        goal = (s.result == "SUCCESS").values
        pt.scatter(D.px(x)[~goal], D.py(y)[~goal], s=40 + scale * s.xg.values[~goal], color=D.alpha(col, 0.35), edgecolors=col,
                   lw=1, ax=ax, zorder=3)
        pt.scatter(D.px(x)[goal], D.py(y)[goal], s=40 + scale * s.xg.values[goal], color=col, edgecolors=style.INK, lw=1.4,
                   ax=ax, zorder=4)
        if label_goals:
            for xx, yy, n in zip(np.asarray(x)[goal], np.asarray(y)[goal], s.player_name.values[goal]):
                ax.annotate(D.short_name(n), (D.px(xx), D.py(yy)), xytext=(6, 6), textcoords="offset points", fontsize=7,
                            color=style.INK, zorder=5)
    return pt


def shot_map_half(ax, M, team, scale=1100):
    pt = D.pitch(ax, half=True)
    s = M.shots[M.shots.team == team]
    col = D.side_col(team == M.home)
    goal = (s.result == "SUCCESS").values
    pt.scatter(D.px(s.x)[~goal], D.py(s.y)[~goal], s=40 + scale * s.xg.values[~goal], color=D.alpha(col, 0.35), edgecolors=col, lw=1,
               ax=ax, zorder=3)
    pt.scatter(D.px(s.x)[goal], D.py(s.y)[goal], s=40 + scale * s.xg.values[goal], color=col, edgecolors=style.INK, lw=1.4, ax=ax,
               zorder=4)
    for xx, yy, n in zip(s.x.values[goal], s.y.values[goal], s.player_name.values[goal]):
        pt.annotate(D.short_name(n), xy=(D.px(xx), D.py(yy)), xytext=(D.px(xx) - 2.5, D.py(yy) + 2), ax=ax, fontsize=7.2,
                    color=style.INK, zorder=5)
    return pt


def key_rows(M, keys):
    return [(TEAM_METRICS[k][0], float(M.hs[k]), float(M.as_[k]), TEAM_METRICS[k][1]) for k in keys]


# ------------------------------------------------------------------ short form
def short_form(R, M):
    p = R.page(M.headline(), M.subline(), portrait=True)
    scoreboard(p, M, 1.75, 1.45)
    W = p.w
    p.section("xG race", top=3.5)
    ax = p.axes(top=3.7, bottom=11 - 5.6, left=0.75, right=0.4)
    xg_race(ax, M)
    p.section("Shot map", top=6.0)
    ax = p.axes(top=6.25, bottom=11 - 8.55, left=0.4, right=W / 2 + 0.05)
    shot_map_full(ax, M, scale=600)
    p.section("Key numbers", top=6.0, left=W / 2 + 0.25)
    ax = p.axes(top=6.25, bottom=11 - 8.75, left=W / 2 + 0.25, right=0.45)
    D.compare_bars(ax, key_rows(M, ["xg", "shots", "on_target", "xdanger", "possession", "pass_pct", "box_entries", "field_tilt"]),
                   M.home, M.away)
    p.section("Momentum: xDanger created per 5 minutes", top=8.95)
    ax = p.axes(top=9.2, bottom=0.95, left=0.55, right=0.45)
    momentum(ax, M)


# ------------------------------------------------------------------ long form pages
def pg_cover(R, M, contents):
    p = R.page(M.headline(), M.subline())
    scoreboard(p, M, 1.75, 1.55)
    p.section("Match facts", top=3.65)
    ax = p.axes(top=3.85, bottom=0.95, left=0.55, right=5.7)
    D.axis_off(ax)
    facts = [("Competition", M.comp), ("Round", M.md or "-"), ("Date", f"{M.date:%A %d %B %Y}"),
             ("Formations", f"{M.m.home_formation or '-'}  vs  {M.m.away_formation or '-'}"),
             ("Half-time", f"{int(M.m.home_ht)}-{int(M.m.away_ht)}" if pd.notna(M.m.home_ht) else "-"),
             ("Match ID", f"IMPECT {M.mid}")]
    ax.set_ylim(len(facts), -0.3)
    for i, (k, v) in enumerate(facts):
        D.text(ax, 0, i + 0.4, k, size=8.5, color=style.MUTED2)
        D.text(ax, 0.35, i + 0.4, v, size=8.8)
    p.section("Contents", top=3.65, left=5.75)
    ax = p.axes(top=3.85, bottom=0.95, left=5.75, right=0.55)
    D.axis_off(ax)
    half = (len(contents) + 1) // 2
    ax.set_xlim(0, 1), ax.set_ylim(half, -0.3)
    for i, c in enumerate(contents):
        x, y = (0 if i < half else 0.5), (i % half) + 0.4
        D.text(ax, x, y, f"{i + 2:02d}", size=8.2, color=style.ACCENT, bold=True)
        D.text(ax, x + 0.07, y, c, size=8.4)


def pg_summary(R, M):
    p = R.page("Executive summary", "The match in short, generated from the numbers in this report.")
    hs, as_ = M.hs, M.as_
    a = M.a
    tops = a.groupby(["player_name", "team"]).xd_added.sum().sort_values(ascending=False)
    lines = [f"{M.home} {M.hg}-{M.ag} {M.away}: xG {hs.xg:.2f}-{as_.xg:.2f}, shots {hs.shots}-{as_.shots}, possession "
             f"{hs.possession:.0%}-{1 - hs.possession:.0%}."]
    better = M.home if hs.xg > as_.xg else M.away
    lines.append(f"{better} created the better chances ({max(hs.xg, as_.xg):.2f} xG against {min(hs.xg, as_.xg):.2f}); "
                 f"xDanger created {hs.xdanger:.2f}-{as_.xdanger:.2f}.")
    for team, s in ((M.home, hs), (M.away, as_)):
        pool = M.L.team_match
        best = sorted(((pct_rank(s[k], pool[k], TEAM_METRICS[k][2]), k) for k in STAT_ROWS if k in s and pd.notna(s[k])), reverse=True)
        hi = ", ".join(f"{TEAM_METRICS[k][0].split(' (')[0].lower()} ({q:.0%} pct)" for q, k in best[:2])
        lo = ", ".join(f"{TEAM_METRICS[k][0].split(' (')[0].lower()} ({q:.0%} pct)" for q, k in best[-2:])
        lines.append(f"{team} against every team-match this season: highest on {hi}; lowest on {lo}.")
    for team in (M.home, M.away):
        t = tops.xs(team, level="team") if team in tops.index.get_level_values("team") else pd.Series(dtype=float)
        if len(t):
            lines.append(f"{team}'s biggest creator by xDanger added: {t.index[0]} ({t.iloc[0]:+.2f}), then {t.index[1]} ({t.iloc[1]:+.2f})."
                         if len(t) > 1 else f"{team}'s biggest creator: {t.index[0]} ({t.iloc[0]:+.2f}).")
    lines.append(f"Set pieces: {M.home} {hs.sp_xg:.2f} xG from attacking restarts, {M.away} {as_.sp_xg:.2f}.")
    ot = M.shots.groupby("team").agg(xg=("xg", "sum"), xgot=("xgot", "sum"), goals=("result", lambda s: (s == "SUCCESS").sum()))
    for team in ot.index:
        r = ot.loc[team]
        lines.append(f"{team} finishing: {D.plural(r.goals, 'goal')} from {r.xg:.2f} xG ({r.xgot:.2f} xGOT on target).")
    ax = p.axes(top=1.85, bottom=0.95)
    D.axis_off(ax)
    ax.set_xlim(0, 1), ax.set_ylim(len(lines) + 0.5, 0)
    for i, l in enumerate(lines):
        ax.scatter([0.005], [i + 0.6], s=22, marker="s", color=style.ACCENT)
        D.text(ax, 0.02, i + 0.6, l, size=9.6)


def pg_stats(R, M):
    p = R.page("Team statistics", "Share of the match total for each metric. Values shown for both teams.")
    groups = [("Attack", ["goals", "xg", "shots", "on_target", "xgot", "xdanger", "box_entries"]),
              ("Possession", ["possession", "passes", "pass_pct", "progressive", "final_third", "crosses", "field_tilt"]),
              ("Defence and set pieces", ["regains", "high_regains", "def_actions", "ppda", "sp_xg", "corners", "fouls"])]
    w = (p.w - 1.1 - 2 * 0.4) / 3
    for k, (lab, keys) in enumerate(groups):
        left = 0.55 + k * (w + 0.4)
        p.section(lab, top=1.85, left=left)
        ax = p.axes(top=2.15, bottom=0.95, left=left, right=p.w - left - w)
        D.compare_bars(ax, key_rows(M, keys), M.home, M.away)


def pg_league(R, M):
    p = R.page("Against the league", f"Each team's numbers as a percentile of all {len(M.L.team_match)} {M.comp} team-matches this "
               "season. PPDA and fouls inverted: right = more pressing, fewer fouls.")
    ax = p.axes(top=2.0, bottom=0.95, left=2.6, right=1.7)
    keys = STAT_ROWS
    pool = M.L.team_match
    y = np.arange(len(keys))[::-1]
    for team, col, dy in ((M.home, D.home_col(), 0.12), (M.away, D.away_col(), -0.12)):
        s = M.hs if team == M.home else M.as_
        q = [pct_rank(s[k], pool[k], TEAM_METRICS[k][2]) * 100 for k in keys]
        ax.scatter(q, y + dy, s=46, color=col, zorder=3, label=team)
    for yi, k in zip(y, keys):
        ax.plot([0, 100], [yi, yi], color=style.HAIR, lw=0.5, zorder=1)
        hv, av = M.hs[k], M.as_[k]
        f = TEAM_METRICS[k][1]
        ax.text(103, yi + 0.12, f.format(hv) if pd.notna(hv) else "-", fontsize=7.6, color=D.home_col(), va="center")
        ax.text(113, yi + 0.12, f.format(av) if pd.notna(av) else "-", fontsize=7.6, color=D.away_col(), va="center")
    ax.axvline(50, color=style.MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.set_yticks(y), ax.set_yticklabels([TEAM_METRICS[k][0] for k in keys], fontsize=8.2, color=style.INK)
    ax.set_xlim(0, 100), ax.set_xticks([0, 25, 50, 75, 100]), ax.set_xlabel("Percentile of league team-matches")
    ax.legend(frameon=False, labelcolor=style.MUTED2, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    D.clean(ax, grid=None)


def pg_xg(R, M):
    p = R.page("Expected goals timeline", "Cumulative xG by match minute; large markers are goals. Below: xDanger created per "
               "5 minutes (up: home, down: away).")
    ax = p.axes(top=1.95, bottom=3.5)
    xg_race(ax, M)
    p.section("Momentum: xDanger created per 5 minutes", top=4.55)
    ax = p.axes(top=4.8, bottom=0.95)
    momentum(ax, M)


def pg_shotmaps(R, M):
    p = R.page("Shot maps", "Every shot, sized by xG. Filled markers are goals. Both teams attack upwards.")
    w = (p.w - 1.1 - 0.4) / 2
    for k, team in enumerate((M.home, M.away)):
        left = 0.55 + k * (w + 0.4)
        s = M.shots[M.shots.team == team]
        p.section(f"{team}  ·  {D.plural(len(s), 'shot')}  ·  {s.xg.sum():.2f} xG  ·  {D.plural((s.result == 'SUCCESS').sum(), 'goal')}", top=1.95,
                  left=left)
        ax = p.axes(top=2.2, bottom=0.95, left=left, right=p.w - left - w)
        shot_map_half(ax, M, team)


def pg_shotlist(R, M):
    s = M.shots.sort_values(["period", "t"])
    p = R.page("Shot list", f"All {len(s)} shots in time order: xG before the shot, xGOT after it (on target only).")
    ax = p.axes(top=1.95, bottom=0.95)
    rows, cols = [], []
    for r in s.itertuples():
        out = "Goal" if r.result == "SUCCESS" else ("On target" if r.on_target else "Off target / blocked")
        rows.append([f"{int(np.ceil(r.minute))}'", r.team, r.player_name, str(r.action).replace("_", " ").lower(),
                     str(r.body_part or "").replace("_", " ").lower(), r.xg, r.xgot if r.on_target else np.nan, out])
        c = D.side_col(r.team == M.home)
        cols.append([None] * 5 + [(c, 0.15 + min(0.7, r.xg * 1.5)), None, (style.ACCENT, 0.35) if out == "Goal" else None])
    n = len(rows)
    if n > 30:
        rows, cols = rows[:30], cols[:30]
    D.table(ax, ["Min", "Team", "Player", "Shot", "Body part", "xG", "xGOT", "Outcome"], rows,
            [0.06, 0.2, 0.2, 0.15, 0.1, 0.07, 0.07, 0.15], ["{}", "{}", "{}", "{}", "{}", "{:.2f}", "{:.2f}", "{}"], cols,
            size=7.4 if len(rows) > 20 else 8)


def pass_network(M, team):
    a = M.side(team)
    starters = a.groupby("player_id").minute.min().sort_values().index[:11]
    a = a[a.player_id.isin(starters)]
    pos = a[a.in_poss == 1].groupby("player_id").agg(x=("x", "mean"), y=("y", "mean"), n=("idx", "size"),
                                                     name=("player_name", "first"))
    allm = M.a.sort_values("idx").reset_index(drop=True)
    nxt = allm.shift(-1)
    ok = (allm.action_type == "PASS") & (allm.result == "SUCCESS") & (allm.team == team) & (nxt.action_type == "RECEPTION") \
        & (nxt.team == team)
    pairs = pd.DataFrame({"a": allm.player_id[ok].values, "b": nxt.player_id[ok].values})
    pairs = pairs[pairs.a.isin(pos.index) & pairs.b.isin(pos.index)]
    pairs["k"] = [tuple(sorted(x)) for x in zip(pairs.a, pairs.b)]
    edges = pairs.groupby("k").size()
    return pos, edges


def pg_passnet(R, M):
    p = R.page("Passing networks", "Starters at their average position in possession; size = touches, lines = completed "
               "passes between two players (3+). Both teams attack to the right.")
    w = (p.w - 1.1 - 0.4) / 2
    for k, team in enumerate((M.home, M.away)):
        left = 0.55 + k * (w + 0.4)
        p.section(team, top=1.95, left=left, )
        ax = p.axes(top=2.2, bottom=1.6, left=left, right=p.w - left - w)
        pt = D.pitch(ax)
        pos, edges = pass_network(M, team)
        col = D.side_col(team == M.home)
        mx = edges.max() if len(edges) else 1
        for (u, v), n in edges.items():
            if n < 3:
                continue
            pt.lines(D.px(pos.x[u]), D.py(pos.y[u]), D.px(pos.x[v]), D.py(pos.y[v]), lw=0.6 + 5 * n / mx,
                     color=D.alpha(col, 0.25 + 0.6 * n / mx), ax=ax, zorder=2)
        pt.scatter(D.px(pos.x), D.py(pos.y), s=60 + 5 * pos.n, color=col, edgecolors=style.BG, lw=1.5, ax=ax, zorder=3)
        for r in pos.itertuples():
            pt.annotate(D.short_name(r.name), xy=(D.px(r.x), D.py(r.y) - 4.2), ax=ax, ha="center", va="top", fontsize=6.8,
                        color=style.INK, zorder=4)
        s = M.hs if team == M.home else M.as_
        D.text(ax, 0, -0.06, f"{int(s.passes)} passes  ·  {s.pass_pct:.0%} completed  ·  {int(s.progressive)} progressive",
               size=8, color=style.MUTED2, transform=ax.transAxes)


def pg_territory(R, M):
    p = R.page("Territory", "Where each team had the ball: touches in possession, smoothed. Both teams attack to the right.")
    w = (p.w - 1.1 - 0.4) / 2
    for k, team in enumerate((M.home, M.away)):
        left = 0.55 + k * (w + 0.4)
        s = M.hs if team == M.home else M.as_
        p.section(f"{team}  ·  field tilt {s.field_tilt:.0%}", top=1.95, left=left)
        ax = p.axes(top=2.2, bottom=1.6, left=left, right=p.w - left - w)
        pt = D.pitch(ax)
        a = M.side(team)
        a = a[a.in_poss == 1]
        pt.kdeplot(D.px(a.x), D.py(a.y), ax=ax, fill=True, levels=40, thresh=0.05, cmap=D.cmap_to(D.side_col(team == M.home)),
                   zorder=1)
        thirds = [(a.x < -17.5).mean(), ((a.x >= -17.5) & (a.x <= 17.5)).mean(), (a.x > 17.5).mean()]
        D.text(ax, 0, -0.06, "Own third {:.0%}  ·  middle {:.0%}  ·  final third {:.0%}".format(*thirds), size=8,
               color=style.MUTED2, transform=ax.transAxes)


def pg_danger(R, M):
    p = R.page("Where the danger came from", "Actions that added the most xDanger (top 25 per team), arrow from start to end; "
               "colour by size of the gain. Both teams attack to the right.")
    w = (p.w - 1.1 - 0.4) / 2
    for k, team in enumerate((M.home, M.away)):
        left = 0.55 + k * (w + 0.4)
        a = M.side(team)
        s = M.hs if team == M.home else M.as_
        p.section(f"{team}  ·  xDanger created {s.xdanger:.2f}  ·  {int(s.box_entries)} box entries", top=1.95, left=left)
        ax = p.axes(top=2.2, bottom=1.6, left=left, right=p.w - left - w)
        pt = D.pitch(ax)
        t = a[a.xd_added > 0].nlargest(25, "xd_added")
        col = D.side_col(team == M.home)
        mx = max(t.xd_added.max(), 1e-6) if len(t) else 1
        for r in t.itertuples():
            pt.arrows(D.px(r.x), D.py(r.y), D.px(r.end_x), D.py(r.end_y), width=1.4, headwidth=4, headlength=4,
                      color=D.alpha(col, 0.3 + 0.7 * r.xd_added / mx), ax=ax, zorder=3)
        ev = a[a.danger_event.isin(["cutback", "golden_zone_touch"])]
        pt.scatter(D.px(ev.x), D.py(ev.y), s=28, color=style.ACCENT, edgecolors=style.BG, ax=ax, zorder=4)
        D.text(ax, 0, -0.06, f"Coral dots: cutbacks and golden-zone touches ({len(ev)})", size=8, color=style.MUTED2,
               transform=ax.transAxes)


def pg_creators(R, M):
    p = R.page("Who created the danger", "xDanger added by player in this match (passes credited with the reception they set up; "
               "shots carry none). xG in brackets.")
    w = (p.w - 1.1 - 0.6) / 2
    for k, team in enumerate((M.home, M.away)):
        left = 0.55 + k * (w + 0.6) + 1.3
        a = M.side(team)
        g = a.groupby("player_name").agg(xd=("xd_added", "sum"), xg=("xg", "sum")).sort_values("xd", ascending=False).head(14)
        p.section(team, top=1.95, left=left - 1.3)
        ax = p.axes(top=2.25, bottom=0.95, left=left, right=p.w - left - w + 1.3)
        D.hbars(ax, [f"{n}  ({x:.2f})" for n, x in zip(g.index, g.xg)], g.xd.values,
                [D.side_col(team == M.home) if v >= 0 else style.GREY for v in g.xd.values], fmt="{:+.2f}", label_size=7.6)


def pg_setpieces(R, M):
    r = M.restarts
    p = R.page("Set pieces and restarts", "Every restart in the match with what followed in the next 30 s of live play: "
               "xG, reaching danger (xDanger 5%+), retention.")
    w = (p.w - 1.1 - 0.4) / 2
    order = ["Corner", "Short corner", "Indirect free kick", "Direct free kick", "Penalty", "Throw-in", "Goal kick", "Kick-off"]
    for k, team in enumerate((M.home, M.away)):
        left = 0.55 + k * (w + 0.4)
        t = r[r.team == team]
        att = t[t.attacking]
        p.section(f"{team}  ·  set-piece xG {att.xg_for.sum():.2f}  ·  goals {int(att.goals_for.sum())}", top=1.95, left=left)
        ax = p.axes(top=2.25, bottom=3.9, left=left, right=p.w - left - w)
        g = t.groupby("restart").agg(n=("xg_for", "size"), xg=("xg_for", "sum"), dang=("reach_danger", "mean"),
                                     keep=("retention", "mean"), goals=("goals_for", "sum")).reindex([o for o in order if o in set(t.restart)])
        D.table(ax, ["Restart", "N", "xG", "Danger", "Kept", "Goals"],
                [[i, rr.n, rr.xg, rr.dang, rr.keep, rr.goals] for i, rr in g.iterrows()],
                [0.38, 0.1, 0.13, 0.13, 0.13, 0.13], ["{}", "{:.0f}", "{:.2f}", "{:.0%}", "{:.0%}", "{:.0f}"], size=8)
        p.section("Corners: where they landed (folded to one flag)", top=4.15, left=left)
        ax = p.axes(top=4.4, bottom=0.95, left=left + 0.8, right=p.w - left - w + 0.8)
        pt = D.pitch(ax, half=True)
        co = t[t.restart == "Corner"]
        fy = np.where(co.y >= 0, co.end_y, -co.end_y)
        col = D.side_col(team == M.home)
        pt.scatter(D.px(co.end_x), D.py(fy), s=40 + 900 * co.xg_for, color=D.alpha(col, 0.5), edgecolors=col, ax=ax, zorder=3)
        pt.scatter(D.px(52.5), D.py(34), s=60, color=style.ACCENT, ax=ax, zorder=4)


def pg_players(R, M, team):
    a = M.side(team)
    pm = M.L.player_match[(M.L.player_match.match_id == M.mid) & (M.L.player_match.team == team)].sort_values("minutes", ascending=False)
    p = R.page(f"{team} players", "Every player who played: minutes (estimated from touches), touches, passing, xDanger added and "
               "received, xG, defensive work. Shading: team best.")
    ax = p.axes(top=1.95, bottom=0.95)
    keys = ["minutes", "touches", "passes", "pass_pct", "progressive", "box_entries", "shots", "xg", "xd_added", "xd_received",
            "regains", "def_actions"]
    fm = ["{:.0f}", "{:.0f}", "{:.0f}", "{:.0%}", "{:.0f}", "{:.0f}", "{:.0f}", "{:.2f}", "{:+.2f}", "{:.2f}", "{:.0f}", "{:.0f}"]
    rows, cols = [], []
    col = D.side_col(team == M.home)
    for r in pm.itertuples():
        rows.append([f"{r.player}  ({r.pos})"] + [getattr(r, k) for k in keys])
        cl = [None]
        for k in keys:
            v, mx = getattr(r, k), pm[k].max()
            cl.append((col, 0.12 + 0.6 * max(0, v) / mx) if k not in ("minutes", "pass_pct") and mx and pd.notna(v) and v > 0 else None)
        cols.append(cl)
    D.table(ax, ["Player", "Min", "Touch", "Pass", "Pass %", "Prog", "Box ent.", "Shots", "xG", "xD +", "xD rec.", "Regain",
                 "Def."], rows, [0.25] + [0.0625] * 12, ["{}"] + fm, cols, size=7.6 if len(rows) > 16 else 8.2)


# ------------------------------------------------------------------ build
def build(comp, match=None, team=None, forms=("short", "long"), out_root=None):
    from .. import config
    M = Match(comp, match, team)
    from pathlib import Path
    folder = Path(out_root or config.OUT_ROOT) / comp / "reports" / "post-match" / slug(f"{M.date:%Y-%m-%d} {M.home} vs {M.away}")
    out = []
    tag = f"{style.THEME}_{style.FONT}"
    if "short" in forms:
        R = Report("post_match", comp, M.eyebrow("Match report"))
        short_form(R, M)
        out.append(R.save(folder, f"short_{tag}"))
    if "long" in forms:
        R = Report("post_match", comp, M.eyebrow("Match report"))
        contents = ["Executive summary", "Team statistics", "Against the league", "Expected goals timeline", "Shot maps",
                    "Shot list", "Passing networks", "Territory", "Where the danger came from", "Who created the danger",
                    "Set pieces and restarts", f"{M.home} players", f"{M.away} players"]
        pg_cover(R, M, contents)
        for fn in (pg_summary, pg_stats, pg_league, pg_xg, pg_shotmaps, pg_shotlist, pg_passnet, pg_territory, pg_danger,
                   pg_creators, pg_setpieces):
            fn(R, M)
        pg_players(R, M, M.home)
        pg_players(R, M, M.away)
        out.append(R.save(folder, f"long_{tag}"))
    return out
