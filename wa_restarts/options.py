"""Ways to create threat from each restart: options per restart type, scored on the models and the five pillars.

Ported from xDanger 33_restart_options.py --competition: one league-season, every value per restart within the same
30 s live window; "vs type" compares an option with the average for its restart type and the "all leagues" columns
with the same option across every IMPECT league (reference/restart_options_all_leagues.csv).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .names import squad_names
from .style import ACCENT, BG, BLUE, GREY, HAIR, INK, MUTED, MUTED2, Page, clean

CHARTS = Path(".")
SRC = "Data: IMPECT  ·  models trained on all IMPECT leagues 2024 to 2027  ·  Waltzing Analytics"
MIN_N = 60
SCOPE = ""
COLS = [("share", "Share", "{:.0%}"), ("reach_danger", "Reach\ndanger", "{:.0%}"), ("peak_xshot", "Peak\nxShot", "{:.0%}"),
        ("xg", "xG per\nrestart", "{:.3f}"), ("goal", "Goal", "{:.1%}"), ("set_piece", "Another\nset piece", "{:.0%}"),
        ("retention", "Retention", "{:.0%}"), ("dominance", "Dominance", "{:.0%}"), ("second_ball", "Second\nball", "{:.0%}"),
        ("goal_against", "Goal\nagainst", "{:.1%}")]


def load(comp):
    ws = config.Workspace(comp)
    f = pd.read_parquet(ws.four_models)
    e = pd.read_parquet(ws.events, columns=["match_id", "idx", "squad_id", "action_type", "action", "x", "y", "end_x",
                                            "end_y", "body_part", "set_piece"])
    e = e[e.action_type.isin(["CORNER", "FREE_KICK", "THROW_IN", "GOAL_KICK", "KICK_OFF"]) | e.action.isin(["DIRECT_FREE_KICK", "PENALTY_KICK"])]
    d = f.merge(e.drop(columns="set_piece").assign(competition=comp), on=["competition", "match_id", "idx"], how="inner")
    sig = pd.read_parquet(ws.possessions, columns=["competition", "match_id", "idx", "signature", "setup_s"])
    d = d.merge(sig, on=["competition", "match_id", "idx"], how="left")
    shots = pd.read_parquet(ws.shots, columns=["competition", "match_id", "idx", "xg", "xgot", "goal",
                                               "on_target", "target_y", "target_z"])
    d = d.merge(shots.rename(columns={"xg": "own_xg", "xgot": "own_xgot", "goal": "own_goal", "on_target": "own_on_target"}),
                on=["competition", "match_id", "idx"], how="left")
    d["reach_danger"] = (d.peak_xd_for >= 0.05).astype(float)
    d["peak_xshot"] = d.peak_xshot_for
    d["xg"] = d.xg_for
    d["goal"] = (d.goals_for > 0).astype(float)
    d["goal_against"] = (d.goals_against > 0).astype(float)
    d["length"] = np.hypot(d.end_x - d.x, d.end_y - d.y)
    d["dx"] = d.end_x - d.x
    d["in_box_end"] = (d.end_x >= 36) & (d.end_y.abs() <= 20.16)
    sign = np.where(d.y >= 0, 1.0, -1.0)
    d["ly"] = d.end_y * sign  # corners folded to the same flag
    return d


def classify(d):
    at, act = d.action_type, d.action
    opt = pd.Series(None, index=d.index, dtype=object)
    typ = pd.Series(None, index=d.index, dtype=object)

    # goal kicks
    m = at == "GOAL_KICK"
    typ[m] = "Goal kick"
    opt[m] = np.select([d.end_x < -36, d.end_x < -17.5, d.end_x < 0, d.end_y.abs() <= 20.16],
                       ["Short, inside own box", "Short, outside the box", "Mid-range, own half", "Long, central"], "Long, wide channel")[m]
    # kick-offs
    m = at == "KICK_OFF"
    typ[m] = "Kick-off"
    opt[m] = np.select([d.end_x < -25, d.end_x < -10, d.end_x <= 10, d.end_y.abs() > 15],
                       ["Back to the defence", "Back to midfield", "Short, sideways", "Long into a channel"], "Long, central")[m]
    # corners: direct deliveries by zone and swing, short corners by what follows
    m = at == "CORNER"
    short = m & ((d.end_x < 36) | (d.ly.abs() > 20.16))
    right = d.body_part == "FOOT_RIGHT"
    swing = np.where((d.y >= 0) & right | (d.y < 0) & ~right, "Inswing", "Outswing")
    zone = np.select([(d.end_x >= 47) & (d.ly > 3.66), (d.end_x >= 47) & (d.ly >= -3.66), d.end_x >= 47,
                      d.end_x >= 41.5], ["near post", "six-yard centre", "far post", "penalty spot"], "edge of the box")
    typ[m & ~short] = "Corner"
    opt[m & ~short] = (pd.Series(swing, index=d.index) + ", " + pd.Series(zone, index=d.index))[m & ~short]
    typ[short] = "Short corner"
    s = d.signature.fillna("")
    rest = s.str.split(" > ").str[1:].str.join(" > ")
    opt[short] = np.select([rest.str.contains("Ball into box"), rest.str.contains("Carry into box"), rest.str.contains("Back pass"),
                            rest.str.contains("Shot")], ["Then cross into the box", "Then carry into the box", "Then recycle back",
                                                         "Then shot from outside"], "Then other / lost")[short]
    # indirect free kicks
    m = (at == "FREE_KICK")
    typ[m] = "Indirect free kick"
    opt[m] = np.select([(d.x > 0) & d.in_box_end & (d.y.abs() > 20.16), (d.x > 0) & d.in_box_end, (d.x > 0) & (d.length < 15),
                        d.x > 0, d.length >= 30], ["Opp. half: wide delivery into box", "Opp. half: central delivery into box",
                                                   "Opp. half: short pass", "Opp. half: other pass", "Own half: long ball"],
                       "Own half: short pass")[m]
    # direct free kicks
    m = act == "DIRECT_FREE_KICK"
    dist = np.hypot(52.5 - d.x, d.y)
    typ[m] = "Direct free kick"
    band = np.select([dist < 20, dist < 25, dist < 30], ["under 20 m", "20 to 25 m", "25 to 30 m"], "30 m +")
    side = np.where(d.y.abs() <= 9.16, "central", "angled")
    opt[m] = (pd.Series(band, index=d.index) + ", " + pd.Series(side, index=d.index))[m]
    # penalties by placement
    m = act == "PENALTY_KICK"
    typ[m] = "Penalty"
    natural = np.where(d.body_part == "FOOT_LEFT", -1, 1)  # mirror left-footers so one side is each taker's natural side
    ty = d.target_y * natural
    height = np.where(d.target_z > 1.2, "high", "low")
    m_pk = (act == "PENALTY_KICK") & d.target_y.notna()
    nat = 1 if (ty[m_pk] > 1.2).sum() >= (ty[m_pk] < -1.2).sum() else -1  # natural side = the side most takers choose
    col = np.select([ty * nat > 1.2, ty * nat >= -1.2], ["Natural side", "Centre"], "Opposite side")
    on = d.own_on_target.fillna(False).astype(bool) | (d.own_goal == 1)
    opt[m] = np.where(~on & ((d.target_y.abs() > 3.66) | (d.target_z > 2.44)), "Off target",
                      pd.Series(col, index=d.index) + ", " + pd.Series(height, index=d.index))[m]
    # throw-ins
    m = at == "THROW_IN"
    typ[m] = "Throw-in"
    ft = d.x > 17.5
    opt[m] = np.select([ft & d.in_box_end, ft & (d.dx < -3), ft & (d.dx > 5), ft, (d.length >= 25) & (d.dx > 5),
                        d.dx > 5, d.dx < -3], ["Final third: long throw into box", "Final third: backward",
                                               "Final third: forward down the line", "Final third: short infield",
                                               "Elsewhere: long forward", "Elsewhere: forward down the line",
                                               "Elsewhere: backward"], "Elsewhere: short infield")[m]
    d["restart"], d["option"] = typ, opt
    return d[d.restart.notna() & d.option.notna()]


def option_table(d):
    rows = []
    for (rt, op), g in d.groupby(["restart", "option"]):
        rows.append({"restart": rt, "option": op, "n": len(g), "reach_danger": g.reach_danger.mean(), "peak_xshot": g.peak_xshot.mean(),
                     "xg": g.xg.mean(), "xg_se": g.xg.std() / np.sqrt(len(g)), "xgot": g.xgot_for.mean(), "goal": g.goal.mean(),
                     "set_piece": g.set_piece.mean(), "retention": g.retention.mean(), "dominance": g.dominance.mean(),
                     "second_ball": g.second_ball.mean(), "goal_against": g.goal_against.mean(), "xg_against": g.xg_against.mean(),
                     "own_xg": g.own_xg.mean(), "own_on_target": g.own_on_target.astype(float).mean(), "own_goal": g.own_goal.mean()})
    t = pd.DataFrame(rows)
    t["share"] = t.n / t.groupby("restart").n.transform("sum")
    tot = d.groupby("restart")[["xg", "goal", "retention", "reach_danger"]].mean().add_prefix("type_")
    t = t.join(tot, on="restart")
    t["xg_vs_type"] = t.xg / t.type_xg
    return t


# ------------------------------------------------------------------ charts
def min_n_for(rt):
    """Direct free kicks are rare: in a single league-season use half the usual minimum."""
    return MIN_N // 2 if rt == "Direct free kick" and MIN_N < 300 else MIN_N


def option_chart(t, rt, order_by="xg", note="", file=None):
    min_n = min_n_for(rt)
    g = t[(t.restart == rt) & (t.n >= min_n)].sort_values(order_by, ascending=False)
    if g.empty:
        return
    best = g.iloc[0]
    if order_by == "own_goal":
        title = f"{rt}: '{best.option}' scores most ({best.own_goal:.0%})"
    elif note:
        keep = g.loc[g.retention.idxmax()]
        title = (f"{rt}: '{best.option}' threatens most and keeps the ball best" if keep.option == best.option
                 else f"{rt}: '{best.option}' threatens most, '{keep.option}' keeps the ball best")
    else:
        title = f"{rt}: '{best.option}' creates the most threat"
    p = Page(0.42 * len(g) + 3.2, f"Restart options  ·  {rt}  ·  {SCOPE}", title,
             f"Per restart, 30 s of live play. Coral: above the {rt.lower()} average, blue: below. {min_n}+ cases per option. {note}",
             source=SRC)
    ax = p.axes(top=2.35, bottom=0.9, left=3.3, right=0.55)
    ax.set_xlim(0, len(COLS)), ax.set_ylim(len(g), 0), ax.set_axis_off()
    from matplotlib.patches import Rectangle
    for j, (k, lab, fmt) in enumerate(COLS):
        ax.text(j + 0.5, -0.25, lab.upper(), ha="center", va="bottom", fontsize=7, weight="bold", color=MUTED2, linespacing=1.3)
        col = g[k].values.astype(float)
        mean = np.nansum(g[k] * g.n) / g.n[g[k].notna()].sum() if k != "share" else np.nan
        for i, v in enumerate(col):
            if k == "share" or np.isnan(v):
                fc, a = GREY, 0.35
            else:
                rel = v / mean - 1 if mean else 0
                good = -rel if k == "goal_against" else rel
                fc, a = (ACCENT if good >= 0 else BLUE), 0.12 + 0.75 * min(1, abs(good) / 0.6)
            ax.add_patch(Rectangle((j + 0.04, i + 0.08), 0.92, 0.84, color=fc, alpha=a, lw=0))
            ax.text(j + 0.5, i + 0.5, "" if np.isnan(v) else fmt.format(v), ha="center", va="center", fontsize=8.3, color=INK)
    for i, r in enumerate(g.itertuples()):
        ax.text(-0.08, i + 0.36, r.option, ha="right", va="center", fontsize=9, color=INK)
        ax.text(-0.08, i + 0.74, f"{r.n:,} restarts", ha="right", va="center", fontsize=6.8, color=MUTED)
    name = file or rt.lower().replace(" ", "_").replace("-", "_")
    p.save(CHARTS / f"O_{name}.png")


def penalty_frame(d, t):
    pk = d[d.restart == "Penalty"]
    p = Page(6.0, f"Penalties  ·  {SCOPE}",
             f"High penalties to either side score {pk[(pk.option.str.endswith('high')) & ~pk.option.str.startswith('Centre')].own_goal.mean():.0%}, low ones {pk[(pk.option.str.endswith('low')) & ~pk.option.str.startswith('Centre')].own_goal.mean():.0%}",
             f"Conversion by placement, {len(pk):,} penalties, seen from the taker. Natural side: the side most takers of that foot choose. Size: share.",
             source=SRC)
    ax = p.axes(top=1.75, bottom=0.85, left=1.0, right=1.0)
    ax.set_xlim(-5.0, 5.0), ax.set_ylim(-0.45, 3.2), ax.set_aspect("equal"), ax.set_axis_off()
    ax.plot([-3.66, -3.66, 3.66, 3.66], [0, 2.44, 2.44, 0], color=INK, lw=4, solid_capstyle="round")
    ax.plot([-5.5, 5.5], [0, 0], color=HAIR, lw=1)
    for x in (-1.2, 1.2):
        ax.plot([x, x], [0, 2.44], color=HAIR, lw=0.6, ls=(0, (3, 3)))
    ax.plot([-3.66, 3.66], [1.2, 1.2], color=HAIR, lw=0.6, ls=(0, (3, 3)))
    cells = {"Opposite side": -2.43, "Centre": 0, "Natural side": 2.43}
    tt = t[t.restart == "Penalty"].set_index("option")
    for col, cx in cells.items():
        for h, cy in [("low", 0.6), ("high", 1.82)]:
            k = f"{col}, {h}"
            if k not in tt.index:
                continue
            r = tt.loc[k]
            ax.scatter(cx, cy, s=200 + 9000 * r.share, color=ACCENT, alpha=float(np.clip(0.25 + 0.7 * (r.own_goal - 0.6) / 0.35, 0.1, 1)), edgecolor=BG, lw=1.5)
            ax.text(cx, cy + 0.08, f"{r.own_goal:.0%}", ha="center", va="center", fontsize=13, weight="bold", color=INK)
            ax.text(cx, cy - 0.25, f"{r.share:.0%} of pens", ha="center", va="center", fontsize=7.5, color=MUTED2)
    if "Off target" in tt.index:
        ax.text(0, 3.0, f"Off target: {tt.loc['Off target', 'share']:.0%} of penalties", ha="center", fontsize=8.5, color=MUTED2)
    reb = pk[(pk.own_goal == 0) & (pk.goals_for > 0)]
    ax.text(0, -0.3, f"Missed or saved penalties that still became a goal within 30 s: {len(reb) / max((pk.own_goal == 0).sum(), 1):.1%}",
            ha="center", fontsize=8, color=MUTED)
    p.save(CHARTS / "O_penalty_placement.png")


def dfk_map(d):
    from mplsoccer import VerticalPitch
    from matplotlib.colors import Normalize
    from .style import DANGER_CMAP
    f = d[d.restart == "Direct free kick"]
    p = Page(7.4, f"Direct free kicks  ·  {SCOPE}",
             f"Only {f.own_goal.mean():.1%} of direct free kicks score; inside 20 m it is {f[np.hypot(52.5 - f.x, f.y) < 20].own_goal.mean():.0%}",
             f"{len(f):,} direct free kicks. Colour: goal rate from the free kick itself, by location (cells with 150+ attempts).",
             source=SRC)
    pt = VerticalPitch(half=True, pad_bottom=-14, pad_top=2, pitch_type="custom", pitch_length=105, pitch_width=68,
                       pitch_color=BG, line_color=HAIR, linewidth=1, line_zorder=3)
    ax = p.axes(top=1.95, bottom=1.4, left=2.2, right=2.2)
    pt.draw(ax=ax)
    s = pt.bin_statistic(f.x + 52.5, f.y + 34, values=f.own_goal, statistic="mean", bins=(12, 9))
    c = pt.bin_statistic(f.x + 52.5, f.y + 34, statistic="count", bins=(12, 9))
    s["statistic"] = np.where(c["statistic"] >= 150, s["statistic"], np.nan)
    norm = Normalize(0, 0.15)
    pt.heatmap(s, ax=ax, cmap=DANGER_CMAP, norm=norm, edgecolor=BG, lw=0.6)
    from .pitch import colorbar
    colorbar(p, DANGER_CMAP, norm, "Goal rate", fmt="{:.0%}")
    p.save(CHARTS / "O_direct_free_kick_map.png")


def overview(t):
    t = t[(t.n >= t.restart.map(min_n_for)) & (t.option != "Off target")].copy()
    t["v"] = np.where(t.restart == "Penalty", t.own_goal, t.xg)  # penalties: conversion (xG is the same for all)
    best = t.sort_values("v", ascending=False).groupby("restart").head(1).set_index("restart")
    worst = t.sort_values("v").groupby("restart").head(1).set_index("restart")
    order = ["Penalty", "Direct free kick", "Corner", "Short corner", "Indirect free kick", "Throw-in", "Goal kick", "Kick-off"]
    order = [o for o in order if o in best.index]
    p = Page(0.62 * len(order) + 3.0, f"Restart options  ·  overview, {SCOPE}",
             "The best option for each restart, against its weakest",
             "xG per restart over 30 s of live play (penalties: conversion), best and weakest option of each type. Log scale.",
             source=SRC)
    ax = p.axes(top=1.95, bottom=1.0, left=2.4, right=0.9)
    for i, rt in enumerate(order):
        b, w_ = best.loc[rt], worst.loc[rt]
        fmt = "{:.0%}" if rt == "Penalty" else "{:.3f}"
        ax.plot([w_.v, b.v], [i, i], color=HAIR, lw=3, zorder=1)
        ax.scatter(w_.v, i, s=70, color=BLUE, edgecolor=BG, lw=1.2, zorder=3)
        ax.scatter(b.v, i, s=70, color=ACCENT, edgecolor=BG, lw=1.2, zorder=3)
        ax.text(b.v * 1.12, i - 0.14, f"{b.option}  " + fmt.format(b.v), va="center", fontsize=8, color=INK)
        ax.text(b.v * 1.12, i + 0.2, f"weakest: {w_.option}  " + fmt.format(w_.v), va="center", fontsize=7, color=MUTED)
    ax.set_xscale("log")
    ax.set_xlim(0.002, 10)
    ax.set_yticks(range(len(order))), ax.set_yticklabels(order, fontsize=9.5, color=INK)
    ax.set_ylim(len(order) - 0.5, -0.6)
    ax.set_xlabel("xG per restart, log scale (penalties: conversion)")
    ax.set_xticks([0.003, 0.01, 0.03, 0.1, 0.3, 1]), ax.set_xticklabels(["0.003", "0.01", "0.03", "0.1", "0.3", "1"])
    clean(ax)
    p.save(CHARTS / "O_00_overview.png")


TEAM_COLS = [("sp_xg_pm", "Set-piece\nxG / match", "{:.2f}"), ("sp_goals", "Set-piece goals\nincl. pens", "{:.0f}"),
             ("corner_xg", "xG per\ncorner", "{:.3f}"), ("corner_goal_area", "Corners to\ngoal area", "{:.0%}"),
             ("short_corner", "Short\ncorners", "{:.0%}"), ("fk_into_box", "Opp.-half FK\ninto box", "{:.0%}"),
             ("fk_xg", "xG per\nopp.-half FK", "{:.3f}"), ("long_throw", "Final-third\nlong throws", "{:.0%}"),
             ("throw_xg", "xG per final-\nthird throw", "{:.3f}"), ("gk_short", "Short\ngoal kicks", "{:.0%}"),
             ("gk_retention", "Goal-kick\nretention", "{:.0%}")]


def team_table(d, comp):
    d = d.merge(squad_names(comp), on="squad_id", how="left")
    nm = d.groupby("team").match_id.nunique()
    att = d[d.restart.isin(["Corner", "Short corner", "Indirect free kick", "Direct free kick", "Penalty", "Throw-in"])
            & ~((d.restart == "Indirect free kick") & d.option.str.startswith("Own half"))
            & ~((d.restart == "Throw-in") & d.option.str.startswith("Elsewhere"))]
    rows = {}
    for team, g in d.groupby("team"):
        a = att[att.team == team]
        co = g[g.restart.isin(["Corner", "Short corner"])]
        fk = g[(g.restart == "Indirect free kick") & g.option.str.startswith("Opp. half")]
        th = g[(g.restart == "Throw-in") & g.option.str.startswith("Final third")]
        gk = g[g.restart == "Goal kick"]
        rows[team] = {"sp_xg_pm": a.xg_for.sum() / nm[team], "sp_goals": a.goals_for.sum(),
                      "corner_xg": co.xg_for.mean(), "corner_goal_area": co.option.str.contains("six-yard|near post|far post|penalty spot").mean(),
                      "short_corner": (co.restart == "Short corner").mean(), "fk_into_box": fk.option.str.contains("into box").mean(),
                      "fk_xg": fk.xg_for.mean(), "long_throw": th.option.str.contains("long throw").mean(), "throw_xg": th.xg_for.mean(),
                      "gk_short": gk.option.str.startswith("Short").mean(), "gk_retention": gk.retention.mean(),
                      "matches": nm[team], "attacking_restarts": len(a)}
    return pd.DataFrame(rows).T.sort_values("sp_xg_pm", ascending=False)


def team_chart(tt, comp):
    from matplotlib.patches import Rectangle
    p = Page(0.36 * len(tt) + 3.3, f"Set-piece profiles  ·  {comp}",
             f"{tt.index[0]} create the most from set pieces, {tt.index[-1]} the least",
             "Attacking restarts (corners, opp.-half free kicks, final-third throw-ins, penalties). Shading: coral above, blue below the league average.",
             source=SRC)
    ax = p.axes(top=2.35, bottom=0.9, left=2.5, right=0.55)
    ax.set_xlim(0, len(TEAM_COLS)), ax.set_ylim(len(tt), 0), ax.set_axis_off()
    for j, (k, lab, fmt) in enumerate(TEAM_COLS):
        ax.text(j + 0.5, -0.25, lab.upper(), ha="center", va="bottom", fontsize=6.6, weight="bold", color=MUTED2, linespacing=1.3)
        v = tt[k].astype(float).values
        mean = np.nanmean(v)
        for i, val in enumerate(v):
            rel = val / mean - 1 if mean else 0
            ax.add_patch(Rectangle((j + 0.04, i + 0.08), 0.92, 0.84, color=ACCENT if rel >= 0 else BLUE,
                                   alpha=0.12 + 0.75 * min(1, abs(rel) / 0.6), lw=0))
            ax.text(j + 0.5, i + 0.5, "" if np.isnan(val) else fmt.format(val), ha="center", va="center", fontsize=8, color=INK)
    for i, team in enumerate(tt.index):
        ax.text(-0.08, i + 0.5, team, ha="right", va="center", fontsize=8.5, color=INK)
    p.save(CHARTS / "O_team_profiles.png")


def run(comp, out_dir):
    """restart_options.csv, team_set_piece_profiles.csv and the O_* charts for one league-season into out_dir."""
    global SCOPE, CHARTS, SRC
    out_dir = Path(out_dir)
    SCOPE = comp
    SRC = f"Data: IMPECT, {comp}  ·  models trained on all IMPECT leagues 2024 to 2027  ·  Waltzing Analytics"
    CHARTS = out_dir / "charts"
    CHARTS.mkdir(parents=True, exist_ok=True)
    for old in CHARTS.glob("O_*.png"):
        old.unlink()
    d = classify(load(comp))
    t = option_table(d)
    # compare every option with the same option across all leagues
    allt = pd.read_csv(config.table("restart_options_all_leagues.csv"))
    t = t.merge(allt[["restart", "option", "xg", "goal", "retention"]].rename(
        columns={"xg": "xg_all_leagues", "goal": "goal_all_leagues", "retention": "retention_all_leagues"}),
        on=["restart", "option"], how="left")
    t.round(4).to_csv(out_dir / "restart_options.csv", index=False)
    overview(t)
    for rt in ["Corner", "Short corner", "Indirect free kick", "Direct free kick", "Throw-in"]:
        option_chart(t, rt)
    option_chart(t, "Goal kick", note="Deep restart: read retention too.")
    option_chart(t, "Kick-off", note="Deep restart: read retention too.")
    option_chart(t, "Penalty", order_by="own_goal", note="Goal includes rebounds.")
    tt = team_table(d, comp)
    tt.to_csv(out_dir / "team_set_piece_profiles.csv")
    team_chart(tt, comp)
    return t, tt
