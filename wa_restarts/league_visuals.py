"""Restart visual pack for one league-season (WA dark navy): pitch maps, scatterplots, pizzas, beeswarms, bars, box plots.

Ported from xDanger 34_league_restart_visuals.py. run(comp, out_dir, focus_team) writes B01 to B24 and the club CSVs.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib.colors import Normalize, to_rgba
from mplsoccer import Pitch, PyPizza, VerticalPitch

from . import config
from . import options as ro
from . import pitch as pv
from .names import squad_names
from .style import ACCENT, BG, BLUE, DANGER_CMAP, GOLD, GREY, HAIR, INK, MUTED, MUTED2, TEAL, L, W, Page, clean

COMP = ""
FOCUS_TEAM = None
OUT = Path(".")
SRC = ""
PITCH = dict(pitch_type="custom", pitch_length=105, pitch_width=68, pitch_color=BG, line_color=HAIR, linewidth=1, line_zorder=3)
ZOOM = dict(half=True, pad_bottom=-24, pad_top=2)
TYPES = ["Corner", "Short corner", "Indirect free kick", "Direct free kick", "Penalty", "Throw-in"]
TCOL = {"Corner": ACCENT, "Short corner": GOLD, "Indirect free kick": TEAL, "Direct free kick": BLUE, "Penalty": "#8C7BD1",
        "Throw-in": "#6B7884"}
N = {"i": 0}


def save(p, name):
    N["i"] += 1
    p.save(OUT / f"B{N['i']:02d}_{name}.png")
    print(f"B{N['i']:02d} {name}", flush=True)


def px(x):
    return np.asarray(x) + 52.5


def py(y):
    return np.asarray(y) + 34


# ------------------------------------------------------------------ data
def load():
    d = ro.classify(ro.load(COMP))
    d = d.merge(squad_names(COMP), on="squad_id", how="left")
    teams = d.groupby("match_id").team.unique()
    d["opponent"] = [next((t for t in teams[m] if t != tm), None) for m, tm in zip(d.match_id, d.team)]
    d["attacking"] = d.restart.isin(TYPES) & ~((d.restart == "Indirect free kick") & d.option.str.startswith("Own half")) \
        & ~((d.restart == "Throw-in") & d.option.str.startswith("Elsewhere"))
    d["fold_y"] = np.where(d.y >= 0, d.end_y, -d.end_y)  # corners folded to the top flag
    curves = pd.read_parquet(config.Workspace(COMP).curves, columns=["competition", "match_id", "idx", "player_name",
                                                                     "delivery_ae", "peak_ae"])
    d = d.merge(curves.drop(columns="competition"), on=["match_id", "idx"], how="left")
    return d


def shots_with_restart(d):
    """Shots by the restart-taking team within 30 s of the last restart before them."""
    s = pd.read_parquet(config.Workspace(COMP).scored, columns=["match_id", "idx", "period", "t", "squad_id", "action_type", "result", "x",
                                                            "y", "xg", "player_name"])
    sh = s[s.action_type == "SHOT"].copy()
    r = d[["match_id", "idx", "squad_id", "restart", "team", "attacking"]].merge(s[["match_id", "idx", "period", "t"]], on=["match_id", "idx"])
    sh, r = sh.sort_values("idx"), r.sort_values("idx")
    m = pd.merge_asof(sh, r.rename(columns={"idx": "r_idx", "squad_id": "r_squad", "t": "r_t", "period": "r_period"}),
                      left_on="idx", right_on="r_idx", by="match_id", direction="backward")
    m = m[(m.r_squad == m.squad_id) & (m.r_period == m.period) & (m.t - m.r_t <= 30) & m.attacking.astype(bool)]
    m["goal"] = m.result == "SUCCESS"
    return m


def club_table(d):
    nm = d.groupby("team").match_id.nunique()
    att = d[d.attacking]
    rows = {}
    for team, g in d.groupby("team"):
        a = att[att.team == team]
        co = g[g.restart.isin(["Corner", "Short corner"])]
        fk = g[(g.restart == "Indirect free kick") & g.option.str.startswith("Opp. half")]
        th = g[(g.restart == "Throw-in") & g.option.str.startswith("Final third")]
        gk = g[g.restart == "Goal kick"]
        opp = att[att.opponent == team]
        rows[team] = {"sp_xg_pm": a.xg_for.sum() / nm[team], "sp_goals": a.goals_for.sum(), "sp_xg_against_pm": opp.xg_for.sum() / nm[team],
                      "sp_goals_against": opp.goals_for.sum(), "corner_xg": co.xg_for.mean(),
                      "corner_goal_area": co.option.str.contains("six-yard|near post|far post|penalty spot").mean(),
                      "short_corner": (co.restart == "Short corner").mean(), "fk_into_box": fk.option.str.contains("into box").mean(),
                      "fk_xg": fk.xg_for.mean(), "long_throw": th.option.str.contains("long throw").mean(), "throw_xg": th.xg_for.mean(),
                      "gk_short": gk.option.str.startswith("Short").mean(), "gk_retention": gk.retention.mean(),
                      "second_ball": a.second_ball.mean(), "set_piece_won": a.set_piece.mean(), "dominance": a.dominance.mean(),
                      "reach_danger": a.reach_danger.mean(), "peak_xshot": a.peak_xshot.mean(), "matches": nm[team]}
    t = pd.DataFrame(rows).T.astype(float)
    return t.sort_values("sp_xg_pm", ascending=False)


# ------------------------------------------------------------------ pitch maps
def corner_landing_map(d):
    co = d[d.restart == "Corner"]
    p, pt, ax = pv.page(f"Corners to the six-yard centre bring {co[co.option.str.contains('six-yard')].xg_for.mean() / co.xg_for.mean():.1f}x the average corner xG",
                        f"Corner landing spots  ·  {COMP}", f"All {len(co):,} direct corners folded to one flag. Colour: mean xG produced within 30 s, cells with 15+ corners.",
                        kind="half", cbar=True, src=SRC)
    s = pt.bin_statistic(px(co.end_x), py(co.fold_y), values=co.xg_for, statistic="mean", bins=(24, 18))
    c = pt.bin_statistic(px(co.end_x), py(co.fold_y), statistic="count", bins=(24, 18))
    s["statistic"] = np.where(c["statistic"] >= 15, s["statistic"], np.nan)
    norm = Normalize(0, np.nanpercentile(s["statistic"], 95))
    pt.heatmap(s, ax=ax, cmap=DANGER_CMAP, norm=norm, edgecolor=BG, lw=0.6)
    pt.scatter(px(52.5), py(34), s=110, color=ACCENT, edgecolors=BG, ax=ax, zorder=5)
    pv.colorbar(p, DANGER_CMAP, norm, "Mean xG per corner")
    save(p, "corner_landing_xg_map")


def corner_club_grid(d):
    co = d[d.restart == "Corner"]
    order = co.groupby("team").xg_for.mean().sort_values(ascending=False).index.tolist()
    cols, rows = 6, 3
    cw = (W - 2 * L - 5 * 0.15) / cols
    chh = cw * 0.5
    p = Page(2.3 + rows * (chh + 0.45) + 1.2, f"Corner landing spots by club  ·  {COMP}",
             f"{order[0]} get the most out of their corners",
             "Every direct corner, folded to one flag. Size and colour: xG produced within 30 s. Number: xG per corner. Sorted by xG per corner.",
             source=SRC)
    pt = VerticalPitch(**ZOOM, **{**PITCH, "linewidth": 0.6})
    norm = Normalize(0, 0.25)
    for k, team in enumerate(order):
        r, c = divmod(k, cols)
        left = L + c * (cw + 0.15)
        top = 2.3 + r * (chh + 0.45)
        ax = p.axes(top=top, bottom=p.h - top - chh, left=left, right=W - left - cw)
        pt.draw(ax=ax)
        g = co[co.team == team]
        pt.scatter(px(g.end_x), py(g.fold_y), s=5 + 150 * g.xg_for.clip(0, 0.6), color=DANGER_CMAP(0.3 + 0.7 * norm(g.xg_for.values)),
                   lw=0, alpha=0.85, ax=ax, zorder=3)
        p.bg.text(left / W, 1 - (top - 0.12) / p.h, team, fontsize=7.3, color=INK, weight="bold", va="center")
        p.bg.text((left + cw) / W, 1 - (top - 0.12) / p.h, f"{g.xg_for.mean():.3f}", fontsize=7.3, color=ACCENT, va="center", ha="right")
    pv.colorbar(p, DANGER_CMAP, norm, "xG produced by the corner")
    save(p, "corner_landing_by_club")


def long_throw_map(d):
    th = d[(d.restart == "Throw-in") & d.option.str.startswith("Final third")]
    lt = th[th.option.str.contains("long throw")]
    top = lt.groupby("team").size().sort_values(ascending=False)
    p, pt, ax = pv.page(f"{top.index[0]} and {top.index[1]} throw long the most",
                        f"Long throws  ·  {COMP}", f"All {len(lt):,} final-third long throws into the box. Colour: xG produced within 30 s. Arrows from the touchline to the landing spot.",
                        kind="half", cbar=True, src=SRC)
    norm = Normalize(0, 0.15, clip=True)
    s = lt.sort_values("xg_for")
    for _, r in s.iterrows():
        c = DANGER_CMAP(0.25 + 0.75 * norm(r.xg_for))
        pt.lines(px(r.x), py(r.y), px(r.end_x), py(r.end_y), ax=ax, color=c, lw=0.3 + 1.6 * norm(r.xg_for), alpha=min(1.0, 0.25 + 0.5 * norm(r.xg_for)),
                 comet=True, zorder=2)
    pv.colorbar(p, DANGER_CMAP, norm, "xG produced by the throw")
    save(p, "long_throws_map")


def fk_delivery_map(d):
    fk = d[(d.restart == "Indirect free kick") & d.option.str.contains("into box")]
    p, pt, ax = pv.page(f"Free-kick deliveries into the box: wide {fk[fk.option.str.contains('wide')].xg_for.mean():.3f} xG, central {fk[fk.option.str.contains('central')].xg_for.mean():.3f}",
                        f"Indirect free kicks  ·  {COMP}", f"{len(fk):,} opponent-half free kicks delivered into the box. Comet: from the free kick to the landing spot. Colour: xG within 30 s.",
                        kind="half", cbar=True, src=SRC)
    norm = Normalize(0, 0.15, clip=True)
    for _, r in fk.sort_values("xg_for").iterrows():
        pt.lines(px(r.x), py(r.y), px(r.end_x), py(r.end_y), ax=ax, color=DANGER_CMAP(0.25 + 0.75 * norm(r.xg_for)),
                 lw=0.3 + 1.6 * norm(r.xg_for), alpha=min(1.0, 0.2 + 0.55 * norm(r.xg_for)), comet=True, zorder=2)
    pv.colorbar(p, DANGER_CMAP, norm, "xG produced by the free kick")
    save(p, "free_kick_deliveries_map")


def goal_kick_grid(d):
    gk = d[d.restart == "Goal kick"]
    order = gk.groupby("team").apply(lambda g: g.option.str.startswith("Short").mean(), include_groups=False).sort_values(ascending=False).index
    cols, rows = 6, 3
    cw = (W - 2 * L - 5 * 0.15) / cols
    chh = cw * 68 / 105
    p = Page(2.5 + rows * (chh + 0.45) + 0.9, f"Goal-kick destinations by club  ·  {COMP}",
             f"{order[0]} play the most goal kicks short, {order[-1]} the fewest",
             "Where each goal kick lands (own goal on the left). Teal: kept the ball for 15 s or until the next restart; grey: lost. Number: share played short.",
             source=SRC)
    p.legend([("Kept", TEAL), ("Lost", GREY)], top=1.9, marker="o")
    pt = Pitch(**{**PITCH, "linewidth": 0.6})
    for k, team in enumerate(order):
        r, c = divmod(k, cols)
        left = L + c * (cw + 0.15)
        top = 2.5 + r * (chh + 0.45)
        ax = p.axes(top=top, bottom=p.h - top - chh, left=left, right=W - left - cw)
        pt.draw(ax=ax)
        g = gk[gk.team == team]
        col = np.where((g.retention == 1).values[:, None], np.array([to_rgba(TEAL)]), np.array([to_rgba(GREY)]))
        pt.scatter(px(g.end_x), py(g.end_y), s=4, color=col, lw=0, alpha=0.7, ax=ax, zorder=3)
        p.bg.text(left / W, 1 - (top - 0.12) / p.h, team, fontsize=6.8 if len(team) > 18 else 7.3, color=INK, weight="bold", va="center")
        p.bg.text((left + cw) / W, 1 - (top - 0.12) / p.h, f"{g.option.str.startswith('Short').mean():.0%}", fontsize=7,
                  color=MUTED2, va="center", ha="right")
    save(p, "goal_kick_destinations_by_club")


def shot_map(sh):
    p, pt, ax = pv.page(f"{int(sh.goal.sum())} set-piece goals from {sh.xg.sum():.0f} xG",
                        f"Set-piece shots  ·  {COMP}", f"Every shot within 30 s of an attacking restart or penalty, by the taking team ({len(sh):,}). Size: xG. Filled: goal, ring: no goal.",
                        kind="half", src=SRC, legend=[(t, TCOL[t]) for t in TYPES if t in sh.restart.unique()])
    for t in TYPES:
        g = sh[(sh.restart == t) & ~sh.goal]
        if not g.empty:
            pt.scatter(px(g.x), py(g.y), s=8 + 300 * g.xg, ax=ax, zorder=2, lw=0.8, edgecolors=[to_rgba(TCOL[t], 0.45)] * len(g),
                       color=to_rgba(BG, 0))
    for t in TYPES:
        g = sh[(sh.restart == t) & sh.goal]
        if not g.empty:
            pt.scatter(px(g.x), py(g.y), s=20 + 400 * g.xg, ax=ax, zorder=4, lw=0.8, color=TCOL[t], edgecolors=BG)
    save(p, "set_piece_shot_map")


def dfk_map(d):
    f = d[d.restart == "Direct free kick"]
    goal = (f.own_goal == 1).values
    p, pt, ax = pv.page(f"{int(goal.sum())} direct free-kick goals from {len(f)} attempts",
                        f"Direct free kicks  ·  {COMP}", "Every direct free kick by location. Size: xG of the shot. Filled coral: goal. Teal ring: on target.",
                        kind="half", src=SRC, legend=[("Goal", ACCENT), ("On target", TEAL), ("Off target / blocked", GREY)])
    on = f.own_on_target.fillna(False).astype(bool).values
    col = np.where(goal[:, None], np.array([to_rgba(ACCENT)]), np.where(on[:, None], np.array([to_rgba(TEAL)]), np.array([to_rgba(GREY)])))
    pt.scatter(px(f.x), py(f.y), s=30 + 900 * f.own_xg.fillna(0), color=col, edgecolors=BG, lw=0.8, alpha=0.9, ax=ax, zorder=3)
    save(p, "direct_free_kick_map")


# ------------------------------------------------------------------ scatterplots
def scatter(t, x, y, xlab, ylab, title, sub, name, invert_y=False, diag=False, fmt=None):
    p = Page(8.8, f"Set-piece profiles  ·  {COMP}", title, sub, source=SRC)
    ax = p.axes(top=1.85, bottom=1.15, left=1.1, right=0.6)
    ax.axvline(t[x].mean(), color=HAIR, lw=1, ls=(0, (3, 3)))
    ax.axhline(t[y].mean(), color=HAIR, lw=1, ls=(0, (3, 3)))
    if diag:
        lim = [min(t[x].min(), t[y].min()) * 0.9, max(t[x].max(), t[y].max()) * 1.05]
        ax.plot(lim, lim, color=MUTED, lw=0.8, ls=(0, (2, 2)))
    ax.scatter(t[x], t[y], s=110, color=ACCENT, edgecolor=BG, lw=1.6, zorder=3)
    texts = [ax.text(r[x], r[y], team, fontsize=8.3, color=INK) for team, r in t.iterrows()]
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color=HAIR, lw=0.6))
    if invert_y:
        ax.invert_yaxis()
    if fmt:
        ax.xaxis.set_major_formatter(fmt)
    ax.set_xlabel(xlab), ax.set_ylabel(ylab)
    clean(ax, grid="both")
    save(p, name)


# ------------------------------------------------------------------ pizzas
PIZZA = [("sp_xg_pm", "Set-piece\nxG / match", False), ("sp_goals", "Set-piece\ngoals", False), ("corner_xg", "xG per\ncorner", False),
         ("corner_goal_area", "Corners to\ngoal area", False), ("fk_xg", "xG per\nfree kick", False), ("throw_xg", "xG per\nfinal-third throw", False),
         ("reach_danger", "Restarts\nreaching danger", False), ("second_ball", "Second\nballs won", False), ("set_piece_won", "Another\nset piece", False),
         ("gk_retention", "Goal-kick\nretention", False), ("sp_xg_against_pm", "Set-piece xG\nconceded", True),
         ("sp_goals_against", "Set-piece goals\nconceded", True)]
PCOL = [ACCENT] * 6 + [GOLD] * 4 + [TEAL] * 2


def percentiles(t):
    pc = pd.DataFrame(index=t.index)
    for k, _, inv in PIZZA:
        r = t[k].rank(pct=True, ascending=not inv)
        pc[k] = ((r - 1 / len(t)) / (1 - 1 / len(t)) * 99 + 1).round().clip(1, 99)
    return pc


def draw_pizza(ax, vals, small=False):
    pz = PyPizza(params=[lab for _, lab, _ in PIZZA], background_color=BG, straight_line_color=HAIR, straight_line_lw=0.8,
                 last_circle_color=HAIR, last_circle_lw=1, other_circle_color=HAIR, other_circle_lw=0.5, other_circle_ls="-",
                 inner_circle_size=10 if small else 14)
    pz.make_pizza([int(v) for v in vals], ax=ax, param_location=112, slice_colors=PCOL, value_colors=[BG] * len(PIZZA),
                  value_bck_colors=PCOL, blank_alpha=0.18, color_blank_space="same",
                  kwargs_slices=dict(edgecolor=BG, zorder=2, linewidth=1),
                  kwargs_params=dict(color=MUTED2 if not small else BG, fontsize=7.5 if not small else 1, va="center"),
                  kwargs_values=dict(color=BG, fontsize=8 if not small else 5.5, zorder=3,
                                     bbox=dict(edgecolor=BG, facecolor="none", boxstyle="round,pad=0.15", lw=0.8)))
    if small:
        for txt in pz.get_param_texts() + pz.get_value_texts():
            txt.set_visible(False)


def pizza_grid(t, pc):
    order = t.index.tolist()
    cols, rows = 6, 3
    cw = (W - 2 * L) / cols
    p = Page(2.4 + rows * (cw + 0.3) + 1.0, f"Set-piece pizzas  ·  {COMP}", "Every club's set-piece profile on one page",
             "Percentile within the league (1 to 99). Coral: threat, gold: control and pillars, teal: defending (inverted, high = concedes little).",
             source=SRC)
    p.legend([("Threat", ACCENT), ("Control and pillars", GOLD), ("Defending", TEAL)], top=1.9)
    for k, team in enumerate(order):
        r, c = divmod(k, cols)
        left = L + c * cw
        top = 2.4 + r * (cw + 0.3)
        ax = p.fig.add_axes([(left + 0.12) / W, (p.h - top - cw + 0.15) / p.h, (cw - 0.24) / W, (cw - 0.24) / p.h], projection="polar")
        draw_pizza(ax, pc.loc[team].values, small=True)
        p.bg.text((left + cw / 2) / W, 1 - (top - 0.02) / p.h, team, ha="center", fontsize=7.5, color=INK, weight="bold")
    save(p, "pizza_all_clubs")


def pizza_single(t, pc, team, rank):
    p = Page(9.6, f"Set-piece pizza  ·  {COMP}", f"{team}: {rank} in the league for set-piece xG per match",
             f"{t.loc[team, 'sp_xg_pm']:.2f} set-piece xG and {int(t.loc[team, 'sp_goals'])} set-piece goals (incl. penalties) in {int(t.loc[team, 'matches'])} matches. Percentile within the league.",
             source=SRC)
    p.legend([("Threat", ACCENT), ("Control and pillars", GOLD), ("Defending (inverted)", TEAL)], top=1.9)
    ax = p.fig.add_axes([(W / 2 - 3.3) / W, 0.95 / p.h, 6.6 / W, 6.6 / p.h], projection="polar")
    draw_pizza(ax, pc.loc[team].values)
    save(p, f"pizza_{team.lower().replace(' ', '_').replace('.', '')}")


# ------------------------------------------------------------------ beeswarms
def swarm_positions(vals, width, step=0.07):
    placed, ys = [], []
    for v in vals:
        off, k = 0.0, 0
        while any(abs(v - pv_) < width and abs(off - po) < 0.06 for pv_, po in placed):
            k += 1
            off = (k + 1) // 2 * step * (1 if k % 2 else -1)
        placed.append((v, off))
        ys.append(off)
    return np.array(ys)


def club_beeswarm(t):
    mets = [("sp_xg_pm", "Set-piece xG per match", "{:.2f}"), ("corner_xg", "xG per corner", "{:.3f}"),
            ("fk_xg", "xG per opp.-half free kick", "{:.3f}"), ("throw_xg", "xG per final-third throw", "{:.3f}"),
            ("second_ball", "Second balls won", "{:.0%}"), ("set_piece_won", "Another set piece won", "{:.0%}"),
            ("sp_xg_against_pm", "Set-piece xG conceded per match", "{:.2f}")]
    p = Page(0.85 * len(mets) + 2.9, f"Clubs on every set-piece metric  ·  {COMP}", "Where each club sits on each metric",
             "One dot per club, standardised within each metric (left = lowest). Best club labelled; for xG conceded the best is the lowest.",
             source=SRC)
    ax = p.axes(top=1.95, bottom=1.0, left=2.9, right=0.9)
    for i, (k, lab, fmt) in enumerate(mets):
        z = (t[k] - t[k].mean()) / t[k].std()
        order = z.sort_values()
        ys = swarm_positions(order.values, 0.18)
        best = t[k].idxmin() if k == "sp_xg_against_pm" else t[k].idxmax()
        cols = [ACCENT if team == best else BLUE for team in order.index]
        ax.scatter(order.values, i + ys, s=[90 if team == best else 40 for team in order.index], c=cols, edgecolor=BG, lw=0.8, zorder=3)
        ax.text(z[best], i + 0.28, f"{best}  {fmt.format(t.loc[best, k])}", ha="center", fontsize=7.5, color=INK)
    ax.set_yticks(range(len(mets))), ax.set_yticklabels([m[1] for m in mets], fontsize=9, color=INK)
    ax.set_ylim(len(mets) - 0.4, -0.6)
    ax.axvline(0, color=HAIR, lw=0.8, ls=(0, (3, 3)))
    ax.set_xlabel("Standard deviations from the league average")
    clean(ax)
    save(p, "beeswarm_clubs")


def taker_beeswarm(d):
    co = d[(d.restart == "Corner") & d.delivery_ae.notna()]
    k = co.groupby(["player_name", "team"]).agg(n=("delivery_ae", "size"), v=("delivery_ae", "mean"), xg=("xg_for", "mean")).reset_index()
    k = k[k.n >= 15].sort_values("v")
    p = Page(6.8, f"Corner takers  ·  {COMP}", f"{k.iloc[-1].player_name} delivers the most dangerous corners",
             "Each dot: one taker with 15+ corners. X: danger right after the delivery against the expected danger for that corner spot (pts per corner). Size: corners.",
             source=SRC)
    ax = p.axes(top=1.85, bottom=1.1, left=0.8, right=0.8)
    v = k.v.values * 100
    ys = swarm_positions(v, (v.max() - v.min()) * 0.035, step=0.22)
    ax.scatter(v, ys, s=k.n * 2.5, c=np.where(v > 0, ACCENT, BLUE), edgecolor=BG, lw=0.8, alpha=0.9, zorder=3)
    pos = dict(zip(k.index, zip(v, ys)))
    lo_y, hi_y = min(-1.5, ys.min() - 0.5), max(1.5, ys.max() + 0.5)
    for side, sel in [("top", k.nlargest(6, "v")), ("bottom", k.nsmallest(3, "v"))]:
        for j, (i, r) in enumerate(sel.iterrows()):
            x0, y0 = pos[i]
            ly = hi_y - 0.3 - j * 0.32 if side == "top" else lo_y + 0.3 + j * 0.32
            ax.annotate(f"{r.player_name} ({r.team})  {r.v * 100:+.1f}", xy=(x0, y0), xytext=(x0, ly), fontsize=7.5, color=INK,
                        ha="right" if side == "top" else "left", va="center",
                        arrowprops=dict(arrowstyle="-", color=HAIR, lw=0.6, shrinkA=2, shrinkB=4))
    ax.axvline(0, color=HAIR, lw=1)
    ax.set_yticks([]), ax.set_ylim(lo_y, hi_y)
    ax.set_xlabel("Delivery danger above expected (percentage points per corner)")
    clean(ax)
    save(p, "beeswarm_corner_takers")


# ------------------------------------------------------------------ bars
def stacked_bars(d, value, title, sub, name, per_match=True):
    a = d[d.attacking & d.restart.isin(TYPES)]
    nm = d.groupby("team").match_id.nunique()
    v = a.groupby(["team", "restart"])[value].sum().unstack(fill_value=0)
    if per_match:
        v = v.div(nm, axis=0)
    v = v[[t for t in TYPES if t in v.columns]]
    v = v.loc[v.sum(axis=1).sort_values().index]
    p = Page(0.36 * len(v) + 2.9, f"Set pieces by club  ·  {COMP}", title.format(top=v.index[-1]), sub, source=SRC)
    p.legend([(t, TCOL[t]) for t in v.columns], top=1.9)
    ax = p.axes(top=2.25, bottom=1.0, left=2.4, right=0.9)
    left = np.zeros(len(v))
    for t in v.columns:
        ax.barh(range(len(v)), v[t], left=left, color=TCOL[t], height=0.68, edgecolor=BG, lw=1)
        left += v[t].values
    for i, tot in enumerate(left):
        ax.text(tot + left.max() * 0.01, i, f"{tot:.2f}" if per_match else f"{tot:.0f}", va="center", fontsize=8.3, color=INK, weight="bold")
    ax.set_yticks(range(len(v))), ax.set_yticklabels(v.index, fontsize=8.8, color=INK)
    ax.set_ylim(-0.6, len(v) - 0.4)
    clean(ax)
    save(p, name)


def corner_mix(d):
    co = d[d.restart.isin(["Corner", "Short corner"])].copy()
    co["zone"] = np.where(co.restart == "Short corner", "Short", co.option.str.split(", ").str[1].str.capitalize())
    zones = ["Near post", "Six-yard centre", "Far post", "Penalty spot", "Edge of the box", "Short"]
    zcol = dict(zip(zones, [BLUE, ACCENT, "#8C7BD1", GOLD, GREY, TEAL]))
    sh = co.groupby("team").zone.value_counts(normalize=True).unstack(fill_value=0).reindex(columns=zones, fill_value=0)
    sh = sh.loc[(sh["Six-yard centre"] + sh["Penalty spot"]).sort_values().index]
    p = Page(0.36 * len(sh) + 2.9, f"Corner targets by club  ·  {COMP}", f"{sh.index[-1]} aim at the six-yard centre and penalty spot most",
             "Share of each club's corners by target zone (folded to one flag). Sorted by six-yard centre plus penalty spot.", source=SRC)
    p.legend([(z, zcol[z]) for z in zones], top=1.9)
    ax = p.axes(top=2.25, bottom=1.0, left=2.4, right=0.6)
    left = np.zeros(len(sh))
    for z in zones:
        ax.barh(range(len(sh)), sh[z], left=left, color=zcol[z], height=0.7, edgecolor=BG, lw=1)
        for i, (x0, w) in enumerate(zip(left, sh[z])):
            if w > 0.08:
                ax.text(x0 + w / 2, i, f"{w:.0%}", ha="center", va="center", fontsize=7, color=BG if z != "Edge of the box" else INK)
        left += sh[z].values
    ax.set_yticks(range(len(sh))), ax.set_yticklabels(sh.index, fontsize=8.8, color=INK)
    ax.set_xlim(0, 1), ax.set_ylim(-0.6, len(sh) - 0.4)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    clean(ax, grid=None)
    save(p, "corner_target_mix")


# ------------------------------------------------------------------ box plots
def box_restart_types(d):
    order = ["Corner", "Short corner", "Indirect free kick", "Direct free kick", "Throw-in", "Goal kick", "Kick-off"]
    order = [o for o in order if o in d.restart.unique()]
    p = Page(7.4, f"Danger by restart type  ·  {COMP}", "Corners and direct free kicks reach the most danger; most deep restarts reach almost none",
             "Peak xDanger reached within 30 s of each restart. Box: middle half of restarts, line: median, whiskers: 5th to 95th percentile.",
             source=SRC)
    ax = p.axes(top=1.85, bottom=1.05, left=2.4, right=2.2)
    data = [d[d.restart == o].peak_xd_for.values * 100 for o in order]
    bp = ax.boxplot(data, vert=False, whis=(5, 95), showfliers=False, patch_artist=True, widths=0.55,
                    medianprops=dict(color=INK, lw=2), whiskerprops=dict(color=MUTED2), capprops=dict(color=MUTED2))
    for patch, o in zip(bp["boxes"], order):
        patch.set(facecolor=to_rgba(TCOL.get(o, BLUE), 0.75), edgecolor=BG)
    for i, o in enumerate(order):
        ax.text(np.percentile(data[i], 95) + 0.5, i + 1, f"median {np.median(data[i]):.1f}%  ·  n {len(data[i]):,}", va="center", fontsize=8, color=MUTED2)
    ax.set_yticks(range(1, len(order) + 1)), ax.set_yticklabels(order, fontsize=9.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Peak xDanger in the 30 s after the restart (%)")
    clean(ax)
    save(p, "box_danger_by_restart_type")


def box_clubs(d):
    a = d[d.attacking]
    m = a.groupby(["team", "match_id"]).xg_for.sum().reset_index()
    order = m.groupby("team").xg_for.median().sort_values(ascending=False).index.tolist()
    p = Page(0.36 * len(order) + 2.8, f"Match to match  ·  {COMP}", f"{order[0]} produce the most set-piece xG in a typical match",
             "Set-piece xG per match (attacking restarts and penalties). Box: middle half of matches, line: median, whiskers: 5th to 95th percentile, dots: matches.",
             source=SRC)
    ax = p.axes(top=1.85, bottom=1.0, left=2.4, right=0.9)
    data = [m[m.team == t].xg_for.values for t in order]
    bp = ax.boxplot(data, vert=False, whis=(5, 95), showfliers=False, patch_artist=True, widths=0.6,
                    medianprops=dict(color=INK, lw=2), whiskerprops=dict(color=MUTED2), capprops=dict(color=MUTED2))
    for patch in bp["boxes"]:
        patch.set(facecolor=to_rgba(ACCENT, 0.55), edgecolor=BG)
    rng = np.random.default_rng(1)
    for i, v in enumerate(data):
        ax.scatter(v, i + 1 + rng.uniform(-0.15, 0.15, len(v)), s=8, color=INK, alpha=0.45, zorder=3)
    ax.set_yticks(range(1, len(order) + 1)), ax.set_yticklabels(order, fontsize=8.8, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Set-piece xG in the match")
    clean(ax)
    save(p, "box_set_piece_xg_by_club")


def box_corner_options(d):
    co = d[d.restart == "Corner"]
    order = co.groupby("option").xg_for.mean().sort_values(ascending=False)
    order = order[co.option.value_counts().reindex(order.index) >= 60].index.tolist()
    p = Page(0.42 * len(order) + 2.9, f"Corner options  ·  {COMP}", "The best corner targets also have the widest spread of outcomes",
             "Peak xShot reached within 30 s of each corner, by target zone and swing. Box: middle half, line: median, whiskers: 5th to 95th percentile. 60+ corners.",
             source=SRC)
    ax = p.axes(top=1.85, bottom=1.0, left=2.8, right=0.9)
    data = [co[co.option == o].peak_xshot.values * 100 for o in order]
    bp = ax.boxplot(data, vert=False, whis=(5, 95), showfliers=False, patch_artist=True, widths=0.55,
                    medianprops=dict(color=INK, lw=2), whiskerprops=dict(color=MUTED2), capprops=dict(color=MUTED2))
    for patch, o in zip(bp["boxes"], order):
        patch.set(facecolor=to_rgba(ACCENT if o.startswith("Inswing") else BLUE, 0.7), edgecolor=BG)
    ax.set_yticks(range(1, len(order) + 1)), ax.set_yticklabels(order, fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Peak xShot in the 30 s after the corner (%)")
    p.legend([("Inswing", ACCENT), ("Outswing", BLUE)], top=p.h - 0.75)
    clean(ax)
    save(p, "box_corner_options_xshot")


def run(comp, out_dir, focus_team=None):
    global COMP, FOCUS_TEAM, OUT, SRC
    COMP, FOCUS_TEAM, OUT = comp, focus_team, Path(out_dir)
    SRC = f"Data: IMPECT, {COMP}  ·  xDanger, xShot, xG, xGOT and restart pillars: Waltzing Analytics"
    N["i"] = 0
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("B[0-9][0-9]_*.png"):  # numbering shifts with the focus club, so clear the last run
        old.unlink()
    d = load()
    t = club_table(d)
    t.round(4).to_csv(OUT / "club_set_piece_table.csv")
    sh = shots_with_restart(d)
    print(f"restarts {len(d):,}  set-piece shots {len(sh):,}", flush=True)
    # pitch maps
    corner_landing_map(d)
    corner_club_grid(d)
    long_throw_map(d)
    fk_delivery_map(d)
    goal_kick_grid(d)
    shot_map(sh)
    dfk_map(d)
    # scatterplots
    scatter(t, "sp_xg_pm", "sp_xg_against_pm", "Set-piece xG per match", "Set-piece xG conceded per match",
            f"{t.sp_xg_pm.idxmax()} attack set pieces best; {t.sp_xg_against_pm.idxmin()} defend them best",
            "Attacking restarts and penalties, for and against. Conceded axis inverted: top right is best. Lines: league average.",
            "scatter_attack_vs_defence", invert_y=True)
    scatter(t, "corner_goal_area", "corner_xg", "Share of corners into the goal area", "xG per corner",
            "Aiming corners at the goal area pays off", "Goal area: near post, six-yard centre, far post and penalty spot. Lines: league average.",
            "scatter_corner_targeting", fmt=lambda v, _: f"{v:.0%}")
    t["sp_goals_pm"] = t.sp_goals / t.matches
    t["xg_total"] = t.sp_xg_pm * t.matches
    scatter(t, "xg_total", "sp_goals", "Set-piece xG (season)", "Set-piece goals incl. penalties (season)",
            f"{(t.sp_goals - t.xg_total).idxmax()} scored the most above their set-piece xG",
            "Season totals. Dashed diagonal: goals equal to xG. Above it: scored more than the chances suggest.",
            "scatter_xg_vs_goals", diag=True)
    scatter(t, "long_throw", "throw_xg", "Share of final-third throws thrown long into the box", "xG per final-third throw-in",
            "Long throws drive final-third throw-in threat", "Final-third throw-ins only. Lines: league average.",
            "scatter_long_throws", fmt=lambda v, _: f"{v:.0%}")
    # pizzas
    pc = percentiles(t)
    pc.to_csv(OUT / "club_percentiles.csv")
    pizza_grid(t, pc)
    ords = {1: "1st", 2: "2nd", 3: "3rd"}
    for i, team in enumerate(t.index[:2]):
        pizza_single(t, pc, team, ords[i + 1])
    pizza_single(t, pc, t.index[-1], f"{len(t)}th")
    if FOCUS_TEAM in t.index and FOCUS_TEAM not in set(t.index[:2]) | {t.index[-1]}:
        rank = t.index.get_loc(FOCUS_TEAM) + 1
        suffix = "th" if 10 <= rank % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(rank % 10, "th")
        pizza_single(t, pc, FOCUS_TEAM, f"{rank}{suffix}")
    # beeswarms
    club_beeswarm(t)
    taker_beeswarm(d)
    # bars
    stacked_bars(d, "xg_for", "{top} create the most set-piece xG per match", "Set-piece xG per match by restart type (attacking restarts and penalties).",
                 "bar_set_piece_xg_by_type")
    d["goal_n"] = d.goals_for
    stacked_bars(d, "goal_n", "{top} score the most set-piece goals", "Goals within 30 s of a restart, by restart type (season totals, incl. penalties).",
                 "bar_set_piece_goals_by_type", per_match=False)
    corner_mix(d)
    # box plots
    box_restart_types(d)
    box_clubs(d)
    box_corner_options(d)
    return t
