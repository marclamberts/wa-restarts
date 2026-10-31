"""Restart windows for one league-season, ported from xDanger 14_restart_curves.py, 15_restart_curve_model.py,
20_restart_possessions.py and 31_restart_four_models.py.

The danger curves are compared with the expected curve of the same restart type and 10 m zone. Those expectations come
from the training seasons of all IMPECT leagues (reference/curve_expectations*.parquet, see build_reference.py), so a
single league gives the same values as the full pipeline.
"""
import numpy as np
import pandas as pd

from .. import config

# ---------------------------------------------------------------- danger curves (14)
SECONDS = 31
CURVE_RESTARTS = {"CORNER", "FREE_KICK", "THROW_IN", "GOAL_KICK"}
CURVE_COLS = ["competition", "match_id", "date", "idx", "period", "t", "squad_id", "player_id", "player_name",
        "action_type", "action", "result", "x", "y", "end_x", "end_y", "in_poss", "xg", "xd_after"]


def team_levels(d, team_a):
    """xDanger level per row from team A's and team B's point of view, carried through non-winning touches."""
    sq, inp, val = d.squad_id.values, d.in_poss.values == 1, d.xd_after.values
    att_a = np.where(sq == team_a, inp, ~inp)  # True when team A is the attacking side
    lv_a = np.where(att_a & (sq == team_a), val, np.where(~att_a, 0.0, np.nan))
    lv_b = np.where(~att_a & (sq != team_a), val, np.where(att_a, 0.0, np.nan))
    block = d.match_id.astype(np.int64).values * 10 + d.period.values
    a = pd.Series(lv_a).groupby(block).ffill().fillna(0).values
    b = pd.Series(lv_b).groupby(block).ffill().fillna(0).values
    return a, b


def curves(scored, comp):
    d = scored[CURVE_COLS].sort_values(["match_id", "idx"]).reset_index(drop=True)
    n = len(d)
    team_a = d.groupby("match_id").squad_id.transform("first").values
    lv_a, lv_b = team_levels(d, team_a)

    at, act = d.action_type.values, d.action.values
    is_restart = np.isin(at, list(CURVE_RESTARTS)) | (act == "DIRECT_FREE_KICK")
    stops = is_restart | (at == "KICK_OFF") | (act == "PENALTY_KICK")
    block = d.match_id.astype(np.int64).values * 10 + d.period.values
    new_block = np.r_[True, block[1:] != block[:-1]]
    bid = np.cumsum(new_block) - 1
    block_end = np.r_[np.flatnonzero(new_block)[1:], n][bid]
    comp = bid * 1e6 + d.t.values

    r = np.flatnonzero(is_restart)
    sidx = np.flatnonzero(stops)
    k = np.searchsorted(sidx, r, side="right")
    stop = np.where(k < len(sidx), sidx[np.minimum(k, len(sidx) - 1)], n)
    stop = np.minimum(stop, block_end[r])  # first row that is no longer part of this restart's live play

    taker_is_a = d.squad_id.values[r] == team_a[r]
    # the ball is dead from shortly after the last live action before the stoppage (IMPECT has no ball-out
    # rows in our extract, and the next restart is often taken 20 to 40 seconds later)
    # play is dead 2 s after the first of: a goal, a gap of more than 10 s to the next event,
    # or the last action before the next restart / the end of the period
    t = d.t.values
    gap_next = np.r_[t[1:] - t[:-1], np.inf]
    gap_next[np.r_[new_block[1:], True]] = np.inf
    nxt_stop = np.r_[stops[1:], True]
    dead_after = (at == "SHOT") & (d.result.values == "SUCCESS") | (gap_next > 10) | nxt_stop
    didx = np.flatnonzero(dead_after)
    first_dead = didx[np.minimum(np.searchsorted(didx, r, side="left"), len(didx) - 1)]
    live_end = comp[np.minimum(first_dead, np.maximum(stop - 1, r))] + 2.0
    F = np.zeros((len(r), SECONDS), dtype=np.float32)
    A = np.zeros((len(r), SECONDS), dtype=np.float32)
    for s in range(SECONDS):
        j = np.searchsorted(comp, comp[r] + s, side="right") - 1
        live = (j < stop) & (comp[r] + s <= live_end)
        F[:, s] = np.where(live, np.where(taker_is_a, lv_a[j], lv_b[j]), 0)
        A[:, s] = np.where(live, np.where(taker_is_a, lv_b[j], lv_a[j]), 0)

    # shots and goals by each side inside the 30-second live window (validation targets)
    t_end = np.minimum(comp[r] + SECONDS - 1, live_end)
    shot = at == "SHOT"
    goal = shot & (d.result.values == "SUCCESS")
    sq = d.squad_id.values
    out = d.iloc[r][["competition", "match_id", "date", "idx", "period", "t", "squad_id", "player_id", "player_name",
                     "x", "y", "end_x", "end_y"]].reset_index(drop=True)
    out["restart_type"] = np.where(act[r] == "DIRECT_FREE_KICK", "DIRECT_FREE_KICK", at[r])
    lo = r
    hi = np.searchsorted(comp, t_end, side="right")
    for name, v in [("xg", np.where(shot, d.xg.values, 0.0)), ("goal", goal.astype(float))]:
        cs_a = np.r_[0.0, np.cumsum(np.where(sq == team_a, v, 0.0))]
        cs_b = np.r_[0.0, np.cumsum(np.where(sq != team_a, v, 0.0))]
        s_a, s_b = cs_a[hi] - cs_a[lo], cs_b[hi] - cs_b[lo]
        out[f"{name}_for"] = np.where(taker_is_a, s_a, s_b)
        out[f"{name}_against"] = np.where(taker_is_a, s_b, s_a)
    for s in range(SECONDS):
        out[f"f{s}"] = F[:, s]
    for s in range(SECONDS):
        out[f"a{s}"] = A[:, s]
    out["peak_for"] = F.max(axis=1)
    out["t_peak"] = F.argmax(axis=1)
    out["area_for"] = F.mean(axis=1)
    out["peak_against"] = A.max(axis=1)
    out["area_against"] = A.mean(axis=1)
    out["live_seconds"] = np.clip(live_end - comp[r], 0, 30)
    return out


# ---------------------------------------------------------------- expected curves (15)
F = [f"f{i}" for i in range(31)]
A = [f"a{i}" for i in range(31)]
PHASES = {"first_contact": range(0, 4), "second_phase": range(4, 13), "sustained": range(13, 31)}
KEY = ["restart_type", "xb", "yb"]


def expected(d):
    """Mean curves per restart type and 10 m zone (200+ training restarts), else the type mean; expected peak."""
    zone = pd.read_parquet(config.table("curve_expectations.parquet"))
    typ = pd.read_parquet(config.table("curve_expectations_type.parquet"))
    d = d.copy()
    d["xb"] = (d.x.clip(-52.5, 52.4) // 10).astype(int)
    d["yb"] = (d.y.abs().clip(0, 33.9) // 10).astype(int)
    e = d[KEY].merge(zone[zone.n >= 200].set_index(KEY)[F + A], left_on=KEY, right_index=True, how="left")
    miss = e[F[0]].isna().values
    e.loc[miss, F + A] = typ.set_index("restart_type").loc[d.restart_type[miss], F + A].values
    pk = d[KEY].merge(zone.set_index(KEY)[["peak_for"]], left_on=KEY, right_index=True, how="left").peak_for.values
    pk = np.where(np.isnan(pk), d.restart_type.map(typ.set_index("restart_type").peak_for).values, pk)
    return d, e[F + A].to_numpy(np.float32), pk


def add_values(d, E, exp_peak):
    fv, av = d[F].to_numpy(np.float32), d[A].to_numpy(np.float32)
    ef, ea = E[:, :31], E[:, 31:]
    d["ds_for"], d["ds_against"] = fv.sum(1), av.sum(1)
    d["exp_ds_for"], d["exp_ds_against"] = ef.sum(1), ea.sum(1)
    d["dae"] = d.ds_for - d.exp_ds_for                # danger above expected (attack)
    d["counter_ae"] = d.ds_against - d.exp_ds_against  # counter danger above expected (bad if positive)
    d["net_dae"] = d.dae - d.counter_ae
    for name, idx in PHASES.items():
        idx = list(idx)
        d[f"dae_{name}"] = fv[:, idx].sum(1) - ef[:, idx].sum(1)
    d["delivery_ae"] = fv[:, 0] - ef[:, 0]            # danger right after the delivery vs expected
    d["peak_for"] = fv.max(1)
    d["exp_peak"] = exp_peak
    d["peak_ae"] = d.peak_for - d.exp_peak             # highest danger reached vs expected
    return d


def score_curves(c):
    d, E, pk = expected(c)
    d = add_values(d, E, pk)
    keep = ["competition", "match_id", "date", "idx", "period", "t", "squad_id", "player_id", "player_name", "x", "y",
            "end_x", "end_y", "restart_type", "xg_for", "goal_for", "xg_against", "goal_against", "ds_for", "exp_ds_for",
            "ds_against", "exp_ds_against", "dae", "counter_ae", "net_dae", "dae_first_contact", "dae_second_phase",
            "dae_sustained", "delivery_ae", "peak_for", "exp_peak", "peak_ae", "t_peak", "live_seconds"]
    return d[keep + F + A]


# ---------------------------------------------------------------- possessions (20)
RESTARTS = {"CORNER", "FREE_KICK", "THROW_IN", "GOAL_KICK"}
PASSLIKE = RESTARTS | {"PASS", "KICK_OFF"}
POSS_COLS = ["match_id", "idx", "period", "t", "squad_id", "action_type", "action", "result", "x", "y", "end_x",
        "end_y", "in_poss", "xd_after"]


def in_box(x, y):
    return (x >= 36) & (np.abs(y) <= 20.16)


def classify(d):
    """One label per action, from the taking team's view."""
    at = d.action_type.values
    x, y, ex, ey = d.x.values, d.y.values, d.end_x.values, d.end_y.values
    length = np.hypot(ex - x, ey - y)
    dx = ex - x
    box_end = in_box(ex, ey) & ~in_box(x, y)
    lab = np.select(
        [at == "SHOT",
         (at == "DRIBBLE") & (length >= 5),
         np.isin(at, list(PASSLIKE)) & box_end,
         np.isin(at, list(PASSLIKE)) & in_box(x, y) & in_box(ex, ey),
         np.isin(at, list(PASSLIKE)) & (length >= 30),
         np.isin(at, list(PASSLIKE)) & (dx < -3),
         np.isin(at, list(PASSLIKE)) & (dx > 10),
         np.isin(at, list(PASSLIKE)),
         at == "LOOSE_BALL_REGAIN"],
        ["Shot", "Carry", "Ball into box", "Pass in box", "Long ball", "Back pass", "Forward pass", "Short pass",
         "Second ball"], "")
    lab = np.where((at == "DRIBBLE") & box_end, "Carry into box", lab)  # touch-and-go dribbles (< 5 m) stay unlabelled
    return lab


def possessions(scored, comp):
    d = scored[POSS_COLS].sort_values(["match_id", "idx"]).reset_index(drop=True)
    n = len(d)
    block = d.match_id.astype(np.int64).values * 10 + d.period.values
    new_block = np.r_[True, block[1:] != block[:-1]]
    t = d.t.values
    at, act = d.action_type.values, d.action.values
    is_restart = np.isin(at, list(RESTARTS)) | (act == "DIRECT_FREE_KICK")
    stop = is_restart | (at == "KICK_OFF") | (act == "PENALTY_KICK")
    goal_prev = np.r_[False, ((at == "SHOT") & (d.result.values == "SUCCESS"))[:-1]]
    gap_prev = np.r_[np.inf, t[1:] - t[:-1]]
    gap_prev[new_block] = np.nan
    seg = np.cumsum(new_block | stop | goal_prev | (np.nan_to_num(gap_prev, nan=np.inf) > 10))
    rid = pd.Series(np.where(is_restart, np.arange(n), np.nan)).groupby(seg).ffill().values
    ok = ~np.isnan(rid)
    w = d[ok].copy()
    w["rid"] = rid[ok].astype(int)
    w["dt"] = t[ok] - t[w.rid.values]
    w = w[w.dt <= 30]
    taker = d.squad_id.values[w.rid.values]
    w["own"] = w.squad_id.values == taker
    w["att_own"] = np.where(w.own, w.in_poss == 1, w.in_poss == 0)
    w["lab"] = classify(w)

    # signature: restart label then the taking team's next labelled actions, until a shot or the ball is lost
    lost = ~w.att_own.values & (w.rid.values != w.index.values)
    w["lost"] = lost
    first_lost = w[w.lost].groupby("rid").dt.min()
    w = w.join(first_lost.rename("t_lost"), on="rid")
    keep = w.own & (w.lab != "") & (w.dt <= w.t_lost.fillna(99))
    seq = w[keep].copy()
    seq["k"] = seq.groupby("rid").cumcount()
    seq = seq[seq.k <= 3]
    sig = seq.groupby("rid").lab.agg(list)

    def cut(labels):
        out = []
        for lab in labels:
            out.append(lab)
            if lab == "Shot":
                break
        return " > ".join(out)

    r = np.flatnonzero(is_restart)
    res = pd.DataFrame({"match_id": d.match_id.values[r], "idx": d.idx.values[r]}, index=r)
    res["setup_s"] = np.where(new_block[r], np.nan, gap_prev[r])
    res["signature"] = sig.reindex(r).map(lambda v: cut(v) if isinstance(v, list) else "").values
    res["lost_s"] = first_lost.reindex(r).values
    # passes and box entries before the peak of the taking team's xDanger
    own = w[w.own & (w.att_own)]
    pk = own.loc[own.groupby("rid").xd_after.idxmax()][["rid", "dt"]].set_index("rid").dt
    res["t_peak_s"] = pk.reindex(r).values
    own = own.join(pk.rename("tp"), on="rid")
    before = own[own.dt <= own.tp]
    res["passes_to_peak"] = before[before.action_type.isin(list(PASSLIKE))].groupby("rid").size().reindex(r).fillna(0).values
    res["players_to_peak"] = before.groupby("rid").squad_id.size().reindex(r).fillna(0).values
    res["competition"] = comp
    return res.reset_index(drop=True)


# ---------------------------------------------------------------- four-model windows (31)
FOUR_RESTARTS = {"CORNER", "FREE_KICK", "THROW_IN", "GOAL_KICK", "KICK_OFF"}


def four_models(scored, xshot, shots, comp):
    d = scored[["match_id", "idx", "period", "t", "squad_id", "action_type", "action", "result",
                "x", "in_poss", "xd_after", "xg"]].sort_values(["match_id", "idx"]).reset_index(drop=True)
    x = xshot[["match_id", "idx", "xshot"]]
    d = d.merge(x, on=["match_id", "idx"], how="left").merge(shots[["match_id", "idx", "xgot", "on_target"]], on=["match_id", "idx"], how="left")
    d["on_target"] = d.on_target.astype(float).fillna(0.0)
    d["xgot"] = d.xgot.fillna(0.0)
    n = len(d)
    block = d.match_id.astype(np.int64).values * 10 + d.period.values
    new_block = np.r_[True, block[1:] != block[:-1]]
    t, at, act, sq = d.t.values, d.action_type.values, d.action.values, d.squad_id.values
    is_restart = np.isin(at, list(FOUR_RESTARTS)) | np.isin(act, ["DIRECT_FREE_KICK", "PENALTY_KICK"])
    goal_prev = np.r_[False, ((at == "SHOT") & (d.result.values == "SUCCESS"))[:-1]]
    gap = np.r_[np.inf, t[1:] - t[:-1]]
    seg = np.cumsum(new_block | is_restart | goal_prev | (gap > 10))
    rid = pd.Series(np.where(is_restart, np.arange(n), np.nan)).groupby(seg).ffill().values
    ok = ~np.isnan(rid)
    w = d[ok].copy()
    w["rid"] = rid[ok].astype(int)
    w["dt"] = t[ok] - t[w.rid.values]
    w = w[w.dt <= 30]
    w["own"] = sq[w.index.values] == sq[w.rid.values]
    w["att"] = np.where(w.own, w.in_poss == 1, w.in_poss == 0)  # the taking team is the attacking side
    w["goal"] = (w.action_type == "SHOT") & (w.result == "SUCCESS")
    w["fx"] = np.where(w.own, w.x, -w.x)  # pitch x in the taking team's frame
    rows = {}
    for side, m in [("for", w.own), ("against", ~w.own)]:
        s = w[m]
        g = s.groupby("rid")
        rows[f"shots_{side}"] = (s.action_type == "SHOT").groupby(s.rid).sum().astype(float)
        rows[f"xg_{side}"] = g.xg.sum()
        rows[f"xgot_{side}"] = g.xgot.sum()
        rows[f"on_target_{side}"] = g.on_target.sum()
        rows[f"goals_{side}"] = g.goal.sum().astype(float)
        poss = s[(s.in_poss == 1)]
        rows[f"peak_xd_{side}"] = poss.groupby("rid").xd_after.max()
        rows[f"peak_xshot_{side}"] = poss.groupby("rid").xshot.max()
    # pillars
    rows["dominance"] = (w.own & (w.fx > 17.5)).groupby(w.rid).mean()
    lb = w[(w.action_type == "LOOSE_BALL_REGAIN") & (w.index.values != w.rid.values)]
    first_lb = lb.groupby("rid").head(1).set_index("rid")
    rows["second_ball"] = first_lb.own.astype(float)
    last = w[w.dt <= 15].groupby("rid").tail(1).set_index("rid")
    rows["retention_live"] = last.att.astype(float)
    out = pd.DataFrame(rows).reindex(np.flatnonzero(is_restart))
    fill0 = [c for c in out if c not in ("second_ball", "retention_live")]
    out[fill0] = out[fill0].fillna(0)
    r = out.index.values
    # next stop after the window: whose restart is it, and does it qualify as another set piece?
    seg_end = pd.Series(np.arange(n)).groupby(seg).transform("max").values
    nxt = np.minimum(seg_end[r] + 1, n - 1)
    same_block = (block[nxt] == block[r]) & (seg_end[r] + 1 < n)
    ended_inside = same_block & is_restart[nxt] & (t[seg_end[r]] - t[r] <= 30)
    ours = sq[nxt] == sq[r]
    qual = ((at[nxt] == "CORNER") | (act[nxt] == "PENALTY_KICK") | (act[nxt] == "DIRECT_FREE_KICK")
            | ((at[nxt] == "FREE_KICK") & (d.x.values[nxt] > 0)) | ((at[nxt] == "THROW_IN") & (d.x.values[nxt] > 17.5)))
    out["set_piece"] = (ended_inside & ours & qual).astype(float)
    scored = out.goals_for.values > 0
    ret = np.where(ended_inside, ours & (at[nxt] != "KICK_OFF"), out.retention_live.fillna(0).values > 0.5)
    out["retention"] = np.where(scored, 1.0, ret.astype(float))
    out = out.drop(columns="retention_live")
    out["match_id"] = d.match_id.values[r]
    out["idx"] = d.idx.values[r]
    out["competition"] = comp
    return out.reset_index(drop=True)
