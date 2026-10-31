"""xDanger for every action, player and team in a league-season (same tables as xDanger 04_outputs.py).

xDanger is how dangerous the next 15 seconds are for the team on the ball: the most dangerous moment it is expected to
reach, as a goal probability. A shot counts at its xG; a danger event that needs no shot counts at the share of those
events followed by a goal within 15 s (golden-zone touch, cutback, box entry by pass or carry, free kick within 30 m,
corner, final-third regain).

    import wa_restarts as wr
    wr.xdanger_actions("Eredivisie 2026-2027")     # one row per on-ball action
    wr.xdanger_players("Eredivisie 2026-2027")     # totals and per 90
    wr.xdanger_teams("Eredivisie 2026-2027")       # per match, for and against
    wr.xdanger("Eredivisie 2026-2027")             # writes all three to <out>/<league>/xdanger/
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .names import squad_names

ACTION_COLS = ["competition", "match_id", "date", "period", "t", "idx", "poss_id", "team", "squad_id", "player_id",
               "player_name", "position", "action_type", "action", "result", "phase", "set_piece", "in_poss", "x", "y",
               "end_x", "end_y", "xg", "xdanger", "xd_before", "xd_after", "xd_added", "xd_won", "xd_denied",
               "xd_shot_sel", "xd_received", "xd_rec_control", "danger_event", "xshot", "xs_added"]
P90 = ["xd_added", "xd_won", "xd_denied", "xd_pass", "xd_carry", "xd_other", "xd_shot_selection", "xd_received",
       "xd_rec_control", "xg"]


def actions(comp, refresh=False):
    """Every on-ball action with xG, xDanger (after the action), xDanger added / won / denied, receiver values and xShot."""
    from . import score
    ws = score(comp, refresh=refresh)
    df = pd.read_parquet(ws.scored)
    xs = pd.read_parquet(ws.xshot, columns=["match_id", "idx", "xshot", "xs_added"])
    df = df.merge(xs, on=["match_id", "idx"], how="left")
    sq = squad_names(comp)
    names = dict(zip(sq.squad_id, sq.team))
    df["team"] = df.squad_id.map(names).fillna(df.squad_id.astype(str))
    return df[[c for c in ACTION_COLS if c in df]].sort_values(["match_id", "idx"]).reset_index(drop=True)


def minutes(df):
    """Approximate minutes per player-match from first/last touch, snapped to period edges."""
    g = df.groupby(["match_id", "period"])["t"].agg(ps="min", pe="max")
    p = df.groupby(["match_id", "period", "player_id"])["t"].agg(f="min", l="max").join(g)
    start = np.where(p.f - p.ps < 300, p.ps, p.f)
    end = np.where(p.pe - p.l < 300, p.pe, p.l)
    p["min"] = (end - start) / 60
    return p.groupby(["match_id", "player_id"])["min"].sum()


def players(comp, min_minutes=0, df=None):
    """Per player: xDanger added (split into passes, carries, other), won, denied, shot selection, receiver values,
    danger-event counts, xG; totals and per 90. Sorted by xDanger added per 90."""
    df = actions(comp) if df is None else df
    mins = minutes(df).groupby("player_id").sum()
    # shots carry no xd_added and receptions hand their value to the passer, so the split is
    # passes (incl. the reception they set up), carries, and everything else
    kind = np.select([df.action_type.isin(["PASS", "RECEPTION"]), df.action_type == "DRIBBLE"], ["pass", "carry"], "other")
    piv = df.assign(kind=kind).pivot_table(index="player_id", columns="kind", values="xd_added", aggfunc="sum", fill_value=0)
    agg = df.groupby("player_id").agg(
        player=("player_name", "first"), team=("team", lambda s: s.mode().iat[0]),
        position=("position", lambda s: s.mode().iat[0] if s.notna().any() else None),
        matches=("match_id", "nunique"), actions=("idx", "size"),
        xd_added=("xd_added", "sum"), xd_won=("xd_won", "sum"), xd_denied=("xd_denied", "sum"),
        xd_shot_selection=("xd_shot_sel", "sum"), xd_received=("xd_received", "sum"), xd_rec_control=("xd_rec_control", "sum"),
        receptions=("xd_received", lambda v: int((v > 0).sum())),
        shots=("action_type", lambda s: (s == "SHOT").sum()), xg=("xg", "sum"))
    agg = agg.join(piv.add_prefix("xd_")).join(mins.rename("minutes"))
    for c in ("xd_pass", "xd_carry", "xd_other"):
        if c not in agg:
            agg[c] = 0.0
    ev = df.dropna(subset=["danger_event"]).pivot_table(index="player_id", columns="danger_event", values="idx",
                                                       aggfunc="size", fill_value=0)
    agg = agg.join(ev.add_prefix("n_")).fillna({f"n_{c}": 0 for c in ev.columns})
    agg["competition"] = comp
    p = agg.reset_index()
    for c in P90:
        p[c + "_p90"] = p[c] / p["minutes"].clip(lower=1) * 90
    # receiver metrics are partly team effects: also relative to the team's outfield average (players with 450+ minutes)
    pool = p[(p.minutes >= 450) & (p.position != "GOALKEEPER")]
    for c in ["xd_received_p90", "xd_rec_control_p90"]:
        p = p.join(pool.groupby("team")[c].mean().rename(c + "_team"), on="team")
        p[c.replace("_p90", "_vs_team_p90")] = p[c] - p[c + "_team"]
        p = p.drop(columns=c + "_team")
    n_cols = [c for c in p if c.startswith("n_")]
    p[n_cols] = p[n_cols].fillna(0).astype(int)
    p = p[p.minutes >= min_minutes]
    return p.round(4).sort_values("xd_added_p90", ascending=False).reset_index(drop=True)


def teams(comp, df=None):
    """Per team and match: xDanger for and against (positive in-possession xDanger added, the version that best predicts
    future goals) and xG for and against."""
    df = actions(comp) if df is None else df
    pos = np.where(df["in_poss"] == 1, df["xd_added"].clip(lower=0), 0.0)
    m = df.assign(xd_pos=pos).groupby(["match_id", "team"]).agg(xd=("xd_pos", "sum"), xg=("xg", "sum")).reset_index()
    opp = m.merge(m, on="match_id", suffixes=("", "_opp"))
    opp = opp[opp.team != opp.team_opp]
    t = opp.groupby("team").agg(matches=("match_id", "nunique"), xd_for=("xd", "mean"), xd_against=("xd_opp", "mean"),
                                xg_for=("xg", "mean"), xg_against=("xg_opp", "mean")).reset_index()
    t["xd_diff"] = t.xd_for - t.xd_against
    t["competition"] = comp
    return t.round(4).sort_values("xd_for", ascending=False).reset_index(drop=True)


def run(comp, out_root=None, refresh=False):
    """<out_root>/<comp>/xdanger/: xdanger_actions.parquet, player_xdanger.csv, team_xdanger.csv."""
    df = actions(comp, refresh=refresh)
    folder = Path(out_root or config.OUT_ROOT) / comp / "xdanger"
    folder.mkdir(parents=True, exist_ok=True)
    df.to_parquet(folder / "xdanger_actions.parquet", index=False)
    players(comp, df=df).to_csv(folder / "player_xdanger.csv", index=False)
    teams(comp, df=df).to_csv(folder / "team_xdanger.csv", index=False)
    print(f"xDanger tables in {folder}", flush=True)
    return folder
