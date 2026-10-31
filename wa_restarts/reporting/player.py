"""Player reports: one match (short portrait page + long landscape pages) or the season so far.

League context compares the player with every player-match (match report, 45+ minutes) or player-season (season report,
min_minutes+) in the same position group: centre backs, full backs, midfielders, attacking midfielders, wingers, forwards.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config, style
from . import draw as D
from .base import Report, slug
from .data import PLAYER_METRICS, league, pct_rank
from .post_match import Match

COUNT = ["touches", "passes", "progressive", "prog_carries", "box_entries", "shots", "xg", "xd_added", "xd_received",
         "regains", "def_actions", "xd_denied", "goals"]
PROFILE = [("On the ball", ["touches", "passes", "pass_pct", "progressive", "prog_carries", "box_entries"]),
           ("End product", ["shots", "xg", "xd_added", "xd_received"]),
           ("Out of possession", ["regains", "def_actions", "xd_denied"])]


class Player:
    def __init__(self, comp, name, team=None):
        self.L = L = league(comp)
        self.comp = comp
        r = L.player(name, team)
        self.pid, self.name, self.team = r.player_id, r.player_name, r.team
        self.a = L.actions[L.actions.player_id == self.pid]
        self.pm = L.player_match[L.player_match.player_id == self.pid].merge(
            L.matches[["match_id", "date", "home", "away", "home_goals", "away_goals"]], on="match_id").sort_values("date")
        self.pos = self.pm.pos.mode().iat[0] if len(self.pm) else "-"
        self.group = self.pm.group.mode().iat[0] if len(self.pm) else "Other"

    def season_table(self, min_minutes):
        pm = self.L.player_match
        g = pm.groupby("player_id")
        t = g[COUNT + ["minutes"]].sum()
        t["matches"] = g.match_id.nunique()
        t["player"], t["team"] = g.player.first(), g.team.first()
        t["group"] = g.group.agg(lambda s: s.mode().iat[0])
        pp = self.L.actions[self.L.actions.action_type == "PASS"]
        t["pass_pct"] = pp.assign(ok=pp.result == "SUCCESS").groupby("player_id").ok.mean()
        for k in COUNT:
            t[k + "_p90"] = t[k] / t.minutes.clip(lower=1) * 90
        pool = t[(t.group == self.group) & (t.minutes >= min_minutes)]
        return t, pool


def _key(k, per90):
    return k if (k == "pass_pct" or not per90) else k + "_p90"


def profile_page(R, title, sub, me, pool, per90=True, rank_label="players"):
    p = R.page(title, sub)
    ax = p.axes(top=2.0, bottom=0.95, left=0.55, right=0.55)
    D.axis_off(ax)
    keys = []
    for lab, ks in PROFILE:
        keys.append(("head", lab))
        keys += [("row", k) for k in ks]
    n = len(keys)
    ax.set_xlim(-100, 130), ax.set_ylim(n + 0.8, -0.8)
    for x, h in ((-30, "PER 90" if per90 else "MATCH"), (104, "PCT"), (114, "RANK")):
        D.text(ax, x, -0.5, h, size=7.2, bold=True, color=style.MUTED2, ha="right" if x < 0 else "left")
    for i, (kind, k) in enumerate(keys):
        y = i + 0.5
        if kind == "head":
            D.text(ax, -98, y, k.upper(), size=8, bold=True, color=style.ACCENT)
            continue
        col = _key(k, per90)
        v = me[col]
        q = pct_rank(v, pool[col])
        rank = int((pool[col] > v).sum()) + 1 if pd.notna(v) else None
        D.text(ax, -98, y, PLAYER_METRICS[k][0], size=8.6)
        D.text(ax, -30, y, "-" if pd.isna(v) else ("{:.0%}" if k == "pass_pct" else "{:.2f}" if per90 else PLAYER_METRICS[k][1]).format(v),
               size=8.6, ha="right")
        ax.barh(y, 100, left=0, height=0.42, color=style.CARD)
        if pd.notna(q):
            ax.barh(y, q * 100, left=0, height=0.42, color=style.ACCENT if q >= 0.5 else style.BLUE)
            D.text(ax, 104, y, f"{q * 100:.0f}", size=8.4)
            D.text(ax, 114, y, f"{rank} / {len(pool)}", size=8.0, color=style.MUTED2)
    ax.plot([50, 50], [0, n], color=style.MUTED, lw=0.8, ls=(0, (3, 3)))
    D.text(ax, 0, n + 0.3, f"Percentile among {len(pool)} {rank_label}. Coral: above the median, blue: below.", size=7.6,
           color=style.MUTED)


def touch_maps(p, a, top, bottom, col, left=0.55, right=0.55, heat=True, label=True):
    w = (p.w - left - right - 0.4) / (2 if heat else 1)
    ax = p.axes(top=top, bottom=bottom, left=left, right=p.w - left - w)
    pt = D.pitch(ax)
    pt.scatter(D.px(a.x), D.py(a.y), s=14, color=col, edgecolors=style.BG, lw=0.4, ax=ax, zorder=3, alpha=0.85)
    if heat:
        ax2 = p.axes(top=top, bottom=bottom, left=left + w + 0.4, right=right)
        pt2 = D.pitch(ax2)
        if len(a) >= 5:
            pt2.kdeplot(D.px(a.x), D.py(a.y), ax=ax2, fill=True, levels=50, thresh=0.04, cmap=D.lava(), zorder=1, bw_adjust=0.7)
        return ax, ax2
    return ax, None


def pass_map(ax, a, col):
    pt = D.pitch(ax)
    ps = a[a.action_type == "PASS"]
    ok = ps.result == "SUCCESS"
    sd, ed = np.hypot(52.5 - ps.x, ps.y), np.hypot(52.5 - ps.end_x, ps.end_y)
    prog = ok & ((sd - ed) >= np.maximum(10, 0.25 * sd)) & (ps.end_x > -17.5)
    pt.lines(D.px(ps.x[ok & ~prog]), D.py(ps.y[ok & ~prog]), D.px(ps.end_x[ok & ~prog]), D.py(ps.end_y[ok & ~prog]), lw=1,
             color=D.alpha(style.MUTED2, 0.55), ax=ax, zorder=2)
    pt.arrows(D.px(ps.x[prog]), D.py(ps.y[prog]), D.px(ps.end_x[prog]), D.py(ps.end_y[prog]), width=1.6, headwidth=4,
              headlength=4, color=col, ax=ax, zorder=3)
    pt.lines(D.px(ps.x[~ok]), D.py(ps.y[~ok]), D.px(ps.end_x[~ok]), D.py(ps.end_y[~ok]), lw=1, ls=":", color=style.RED, ax=ax,
             zorder=2)
    return len(ps), int(ok.sum()), int(prog.sum())


def shot_panel(ax, s, col):
    pt = D.pitch(ax, half=True)
    goal = (s.result == "SUCCESS").values
    pt.scatter(D.px(s.x)[~goal], D.py(s.y)[~goal], s=50 + 1100 * s.xg.values[~goal], color=D.alpha(col, 0.35), edgecolors=col,
               ax=ax, zorder=3)
    pt.scatter(D.px(s.x)[goal], D.py(s.y)[goal], s=50 + 1100 * s.xg.values[goal], color=col, edgecolors=style.INK, lw=1.4, ax=ax,
               zorder=4)


def defend_panel(ax, a):
    pt = D.pitch(ax)
    d = a[a.action_type.isin(["INTERCEPTION", "LOOSE_BALL_REGAIN", "BLOCK", "CLEARANCE", "GROUND_DUEL", "FOUL"])]
    for act, col, mk in (("LOOSE_BALL_REGAIN", style.ACCENT, "o"), ("INTERCEPTION", style.GOLD, "D"), ("GROUND_DUEL", style.TEAL, "s"),
                         ("CLEARANCE", style.BLUE, "^"), ("BLOCK", style.PURPLE, "v"), ("FOUL", style.RED, "x")):
        e = d[d.action_type == act]
        pt.scatter(D.px(e.x), D.py(e.y), s=34, color=col, marker=mk, edgecolors=style.BG if mk != "x" else None, lw=0.6, ax=ax,
                   zorder=3, label=act.replace("_", " ").lower())
    ax.legend(frameon=False, labelcolor=style.MUTED2, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.0))


# ------------------------------------------------------------------ match report
def match_report(comp, name, match=None, team=None, forms=("short", "long"), out_root=None):
    P = Player(comp, name, team)
    M = Match(comp, match, P.team)
    if P.pid not in set(M.a.player_id):
        raise ValueError(f"{P.name} did not play in {M.home} vs {M.away}")
    a = M.a[M.a.player_id == P.pid]
    me = M.L.player_match[(M.L.player_match.match_id == M.mid) & (M.L.player_match.player_id == P.pid)].iloc[0]
    pool = M.L.player_match[(M.L.player_match.group == P.group) & (M.L.player_match.minutes >= 45)].copy()
    for k in PLAYER_METRICS:
        if k != "pass_pct":
            pool[k + "_p90"] = pool[k] / pool.minutes.clip(lower=1) * 90
    me = me.copy()
    for k in PLAYER_METRICS:
        if k != "pass_pct":
            me[k + "_p90"] = me[k] / max(me.minutes, 1) * 90
    s = M.shots[M.shots.player_id == P.pid]
    col = D.side_col(P.team == M.home)
    opp = M.away if P.team == M.home else M.home
    head = (f"{P.name}: " + ", ".join(x for x in (D.plural(me.goals, "goal") if me.goals else None,
                                                  D.plural(me.shots, "shot") if me.shots else None,
                                                  f"{me.xd_added:+.2f} xDanger added", f"{int(me.regains)} ball regains") if x))
    sub = f"{P.pos}, {P.team}  ·  {me.minutes:.0f} minutes  ·  {M.home} {M.hg}-{M.ag} {M.away}  ·  {M.date:%d %B %Y}"
    eyebrow = M.eyebrow("Player report")
    folder = Path(out_root or config.OUT_ROOT) / comp / "reports" / "players" / slug(P.name) / slug(f"{M.date:%Y-%m-%d} vs {opp}")
    tag = f"{style.THEME}_{style.FONT}"
    out = []
    mrank = M.L.player_match[M.L.player_match.match_id == M.mid]

    def cards():
        def rk(k):
            return f"rank {int((mrank[k] > me[k]).sum()) + 1} of {len(mrank)} in the match"
        return [("xDanger added", f"{me.xd_added:+.2f}", rk("xd_added")), ("Touches", f"{int(me.touches)}", rk("touches")),
                ("Passes", f"{int(round(me.passes * me.pass_pct if pd.notna(me.pass_pct) else 0))}/{int(me.passes)}",
                 f"{me.pass_pct:.0%} completed" if pd.notna(me.pass_pct) else ""),
                ("Shots  ·  xG", f"{int(me.shots)}  ·  {me.xg:.2f}", D.plural(me.goals, "goal")),
                ("Box entries", f"{int(me.box_entries)}", rk("box_entries")), ("Regains", f"{int(me.regains)}", rk("regains"))]

    if "short" in forms:
        R = Report("player", comp, eyebrow)
        p = R.page(head, sub, portrait=True)
        c = cards()
        p.cards(c[:3], top=1.7, height=0.85)
        p.cards(c[3:], top=2.65, height=0.85)
        W = p.w
        p.section("Touches", top=3.8)
        ax = p.axes(top=4.0, bottom=11 - 6.3, left=0.45, right=W / 2 + 0.1)
        pt = D.pitch(ax)
        pt.scatter(D.px(a.x), D.py(a.y), s=12, color=col, edgecolors=style.BG, lw=0.4, ax=ax, zorder=3)
        p.section("Passes (arrows: progressive, dotted: failed)", top=3.8, left=W / 2 + 0.1)
        ax = p.axes(top=4.0, bottom=11 - 6.3, left=W / 2 + 0.1, right=0.45)
        pass_map(ax, a, col)
        p.section("Shots", top=6.6)
        ax = p.axes(top=6.8, bottom=11 - 9.0, left=0.45, right=W / 2 + 0.1)
        shot_panel(ax, s, col)
        p.section("Defending", top=6.6, left=W / 2 + 0.1)
        ax = p.axes(top=6.8, bottom=11 - 8.75, left=W / 2 + 0.1, right=0.45)
        defend_panel(ax, a)
        p.section("Touches per 5 minutes", top=9.45)
        ax = p.axes(top=9.65, bottom=0.95, left=0.6, right=0.45)
        b = (a.minute // 5 * 5).value_counts().sort_index()
        ax.bar(b.index + 2.5, b.values, width=4.3, color=col)
        ax.set_xlim(0, max(95, M.a.minute.max())), ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
        D.clean(ax, grid="y")
        out.append(R.save(folder, f"short_{tag}"))

    if "long" in forms:
        R = Report("player", comp, eyebrow)
        p = R.page(head, sub)
        c = cards()
        p.cards(c[:3], top=1.85, height=1.0)
        p.cards(c[3:], top=3.0, height=1.0)
        p.section("xDanger added through the match (cumulative)", top=4.35)
        ax = p.axes(top=4.6, bottom=0.95)
        aa = a.sort_values(["period", "t"])
        ax.step(np.r_[0, aa.minute], np.r_[0, aa.xd_added.cumsum()], where="post", color=col, lw=2)
        ax.axhline(0, color=style.HAIR, lw=0.8), ax.set_xlim(0, max(95, M.a.minute.max()))
        ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
        D.clean(ax, grid="y")
        profile_page(R, "Percentile profile", f"Per 90 in this match against every {M.comp} {P.group.lower()} player-match with 45+ "
                     "minutes this season.", me, pool, per90=True, rank_label=f"{P.group.lower()} player-matches")
        p = R.page("Touch map and heatmap", f"{len(a)} on-ball actions (left) and where they cluster (right). Attacking to the right.")
        touch_maps(p, a, 1.95, 1.5, col)
        thirds = [(a.x < -17.5).mean(), ((a.x >= -17.5) & (a.x <= 17.5)).mean(), (a.x > 17.5).mean()]
        p.bg.text(0.55 / p.w, 1.15 / p.h, "Own third {:.0%}  ·  middle third {:.0%}  ·  final third {:.0%}".format(*thirds),
                  fontsize=8.5, color=style.MUTED2)
        p = R.page("Passing", "Every pass: progressive passes as arrows, other completed passes as lines, failed passes dotted.")
        ax = p.axes(top=1.95, bottom=1.3, left=0.55, right=3.4)
        n, ok, prog = pass_map(ax, a, col)
        ax = p.axes(top=2.2, bottom=3.5, left=p.w - 3.0, right=0.55)
        ps = a[a.action_type == "PASS"]
        L_ = np.hypot(ps.end_x - ps.x, ps.end_y - ps.y)
        rows = [["Passes", n], ["Completed", ok], ["Progressive", prog], ["Short (<15 m)", int((L_ < 15).sum())],
                ["Medium (15-30 m)", int(((L_ >= 15) & (L_ < 30)).sum())], ["Long (30 m+)", int((L_ >= 30).sum())],
                ["Forward", int((ps.end_x - ps.x > 5).sum())], ["Backward", int((ps.end_x - ps.x < -5).sum())]]
        D.table(ax, ["", ""], rows, [0.7, 0.3], ["{}", "{:.0f}"])
        p = R.page("Danger created and received", "Left: actions coloured by xDanger added (coral gains, blue losses). "
                   "Right: where passes were received, coloured by the danger of the spot.")
        w = (p.w - 1.1 - 0.4) / 2
        ax = p.axes(top=1.95, bottom=1.3, left=0.55, right=p.w - 0.55 - w)
        pt = D.pitch(ax)
        t = a[a.xd_added.abs() > 0.005]
        pt.scatter(D.px(t.x), D.py(t.y), s=20 + 600 * t.xd_added.abs(), c=np.where(t.xd_added > 0, style.ACCENT, style.BLUE), ax=ax,
                   edgecolors=style.BG, zorder=3)
        ax = p.axes(top=1.95, bottom=1.3, left=0.95 + w, right=0.55)
        pt = D.pitch(ax)
        rc = a[a.xd_received > 0]
        pt.scatter(D.px(rc.x), D.py(rc.y), s=30, c=rc.xd_received, cmap=D.lava(), vmin=0, vmax=max(0.05, rc.xd_received.max()),
                   edgecolors=style.HAIR, lw=0.4, ax=ax, zorder=3)
        p = R.page("Shots and defending", f"{D.plural(len(s), 'shot')} ({s.xg.sum():.2f} xG). Defensive actions by type, attacking to the right.")
        ax = p.axes(top=1.95, bottom=1.3, left=0.55, right=p.w - 0.55 - 4.2)
        shot_panel(ax, s, col)
        ax = p.axes(top=1.95, bottom=1.6, left=5.2, right=0.55)
        defend_panel(ax, a)
        out.append(R.save(folder, f"long_{tag}"))
    return out


# ------------------------------------------------------------------ season report
def season_report(comp, name, team=None, min_minutes=450, out_root=None):
    P = Player(comp, name, team)
    t, pool = P.season_table(min_minutes)
    me = t.loc[P.pid]
    a, pm = P.a, P.pm
    col = style.ACCENT
    head = f"{P.name}: {me.xd_added_p90:+.2f} xDanger added per 90"
    sub = (f"{P.pos}, {P.team}  ·  {D.plural(me.matches, "match", "matches")}, {me.minutes:.0f} minutes  ·  {D.plural(me.goals, "goal")}, {me.xg:.2f} xG  ·  "
           f"{P.comp}")
    R = Report("player_season", comp, f"{comp}  ·  Season report")
    p = R.page(head, sub)
    in_pool = P.pid in pool.index

    def rk(k):
        if not in_pool:
            return f"under {min_minutes} min: not ranked"
        return f"rank {int((pool[k] > me[k]).sum()) + 1} of {len(pool)} {P.group.lower()}"
    c = [("xDanger added p90", f"{me.xd_added_p90:+.2f}", rk("xd_added_p90")), ("xG p90", f"{me.xg_p90:.2f}", rk("xg_p90")),
         ("Box entries p90", f"{me.box_entries_p90:.1f}", rk("box_entries_p90")),
         ("Progressive passes p90", f"{me.progressive_p90:.1f}", rk("progressive_p90")),
         ("Regains p90", f"{me.regains_p90:.1f}", rk("regains_p90")), ("Pass completion", f"{me.pass_pct:.0%}", rk("pass_pct"))]
    p.cards(c[:3], top=1.85, height=1.0)
    p.cards(c[3:], top=3.0, height=1.0)
    p.section("xDanger added and xG by match", top=4.35)
    ax = p.axes(top=4.6, bottom=1.3)
    x = np.arange(len(pm))
    ax.bar(x - 0.2, pm.xd_added, width=0.38, color=style.ACCENT, label="xDanger added")
    ax.bar(x + 0.2, pm.xg, width=0.38, color=style.BLUE, label="xG")
    ax.axhline(0, color=style.HAIR, lw=0.8)
    opp = np.where(pm.home == P.team, "v " + pm.away, "@ " + pm.home)
    ax.set_xticks(x), ax.set_xticklabels([o[:14] for o in opp], rotation=40, ha="right", fontsize=7)
    ax.legend(frameon=False, labelcolor=style.MUTED2, fontsize=8)
    D.clean(ax, grid="y")
    profile_page(R, "Percentile profile", f"Per 90 this season against {len(pool)} {P.comp} {P.group.lower()} with {min_minutes}+ "
                 "minutes.", me, pool, per90=True, rank_label=P.group.lower())
    p = R.page(f"Where {P.name} plays", f"{len(a):,} on-ball actions this season: every touch (left) and where they cluster (right). "
               "Attacking to the right.")
    touch_maps(p, a, 1.95, 1.4, col)
    p = R.page("Progression", "Progressive passes (arrows) and progressive carries (lines) this season.")
    ax = p.axes(top=1.95, bottom=1.3, left=0.55, right=0.55 + 3.6)
    pt = D.pitch(ax)
    sd, ed = np.hypot(52.5 - a.x, a.y), np.hypot(52.5 - a.end_x, a.end_y)
    prog = (a.result == "SUCCESS") & ((sd - ed) >= np.maximum(10, 0.25 * sd)) & (a.end_x > -17.5)
    pp, pc = a[prog & (a.action_type == "PASS")], a[prog & (a.action_type == "DRIBBLE")]
    pt.arrows(D.px(pp.x), D.py(pp.y), D.px(pp.end_x), D.py(pp.end_y), width=1.4, headwidth=4, headlength=4,
              color=D.alpha(style.ACCENT, 0.75), ax=ax, zorder=3)
    pt.lines(D.px(pc.x), D.py(pc.y), D.px(pc.end_x), D.py(pc.end_y), lw=1.6, color=D.alpha(style.TEAL, 0.8), ax=ax, zorder=3)
    ax = p.axes(top=2.2, bottom=3.0, left=p.w - 3.3, right=0.55)
    ev = a.danger_event.value_counts()
    rows = [["Progressive passes", len(pp)], ["Progressive carries", len(pc)]] + \
        [[k.replace("_", " ").capitalize(), int(v)] for k, v in ev.items()]
    D.table(ax, ["", "Season"], rows, [0.75, 0.25], ["{}", "{:.0f}"])
    s = P.L.shots[P.L.shots.player_id == P.pid]
    p = R.page("Shots", f"{D.plural(len(s), 'shot')}, {s.xg.sum():.2f} xG, {D.plural((s.result == 'SUCCESS').sum(), 'goal')}, "
               f"{s.xgot.sum():.2f} xGOT on target. Size: xG; filled: goal.")
    ax = p.axes(top=1.95, bottom=1.0, left=0.55, right=0.55 + 4.4)
    shot_panel(ax, s, col)
    ax = p.axes(top=2.2, bottom=1.2, left=p.w - 4.2, right=0.55)
    rows = [[f"{pd.to_datetime(d):%d %b}", int(np.ceil(m_)), str(act).replace("_", " ").lower(), xg_,
             "Goal" if r_ == "SUCCESS" else ("On target" if ot else "-")]
            for d, m_, act, xg_, r_, ot in zip(s.match_id.map(P.L.matches.set_index("match_id").date), s.minute, s.action, s.xg,
                                               s.result, s.on_target)][:24]
    D.table(ax, ["Date", "Min", "Shot", "xG", "Outcome"], rows, [0.18, 0.1, 0.36, 0.14, 0.22],
            ["{}", "{:.0f}", "{}", "{:.2f}", "{}"], size=7.4)
    p = R.page("Match log", "Every match this season.")
    ax = p.axes(top=1.95, bottom=0.95)
    rows = [[f"{pd.to_datetime(r.date):%d %b}", o, f"{r.home_goals}-{r.away_goals}", r.minutes, r.touches, r.passes, r.pass_pct,
             r.shots, r.xg, r.xd_added, r.box_entries, r.regains] for r, o in zip(pm.itertuples(), opp)][:26]
    D.table(ax, ["Date", "Opponent", "Score", "Min", "Touch", "Pass", "Pass %", "Shots", "xG", "xD +", "Box ent.", "Regain"], rows,
            [0.08, 0.22, 0.07, 0.06, 0.07, 0.07, 0.07, 0.07, 0.07, 0.08, 0.07, 0.07],
            ["{}", "{}", "{}", "{:.0f}", "{:.0f}", "{:.0f}", "{:.0%}", "{:.0f}", "{:.2f}", "{:+.2f}", "{:.0f}", "{:.0f}"], size=7.8)
    folder = Path(out_root or config.OUT_ROOT) / comp / "reports" / "players" / slug(P.name)
    return [R.save(folder, f"season_{style.THEME}_{style.FONT}")]
