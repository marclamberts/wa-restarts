"""Model features and crediting, ported from xDanger 02_xg.py, 03_xdanger.py (events version) and 25_xgot.py."""
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- xDanger / xShot (03_xdanger.py)
NUM = ["x", "y", "end_x", "end_y", "abs_end_y", "start_dist", "start_mouth", "end_dist",
       "end_mouth", "dx", "pass_dist", "pressure", "opponents", "dist_opp_m", "in_poss",
       "set_piece", "p1_same", "p1_end_x", "p1_end_y", "p1_dt", "p2_same", "p2_end_x",
       "p2_end_y", "p2_dt", "poss_t", "poss_n", "poss_start_x"]
CAT = ["action_type", "action", "result", "phase", "body_part", "prev_pass_height",
       "receiver_type", "p1_type", "p2_type"]
ID = ["competition", "match_id", "date", "idx", "period", "t", "squad_id", "player_id",
      "player_name", "position", "action_type", "action", "result", "phase", "set_piece",
      "x", "y", "end_x", "end_y", "poss_id"]
DEFENSIVE = {"INTERCEPTION", "LOOSE_BALL_REGAIN", "BLOCK", "CLEARANCE", "GROUND_DUEL",
             "GK_CATCH", "GK_SAVE"}
RESTARTS = {"CORNER", "FREE_KICK", "THROW_IN", "GOAL_KICK", "KICK_OFF"}
DELIVERIES = {"PASS", "CORNER", "FREE_KICK", "THROW_IN", "GOAL_KICK", "KICK_OFF"}


def xg_for(df, xgm):
    s = df["action_type"] == "SHOT"
    out = np.zeros(len(df))
    if s.any():
        sh = df[s].copy()
        sh["abs_y"] = sh["y"].abs()
        sh["gk_goal_dist"] = np.hypot(52.5 - sh["gk_x"], sh["gk_y"])
        sh["gk_shot_dist"] = np.hypot(sh["gk_x"] - sh["x"], sh["gk_y"] - sh["y"])
        sh["set_piece"] = sh["set_piece"].astype(int)
        for c in xgm["cat"]:
            sh[c] = pd.Categorical(sh[c], categories=xgm["cats"][c])
        out[s.values] = xgm["model"].predict_proba(sh[xgm["num"] + xgm["cat"]])[:, 1]
    return out


def prep(df, cats):
    X = df[NUM + CAT].copy()
    for c in CAT:
        X[c] = pd.Categorical(X[c], categories=cats[c])
    X[NUM] = X[NUM].astype(np.float32)
    return X


def credit(df, after):
    """Split each action's value into added / won / denied using IMPECT possessions (poss_id)."""
    in_poss = df["in_poss"].values == 1
    after_s = pd.Series(after, index=df.index)
    # inside a possession: before = the attacking team's previous action in the same possession,
    # so opponent touches without the ball (blocks, deflections) do not reset it
    att = df[in_poss]
    prev = att.groupby("poss_id").cumcount() > 0
    before = pd.Series(0.0, index=df.index)
    before[att.index] = np.where(prev, after_s[att.index].groupby(att["poss_id"]).shift().fillna(0).values, 0.0)
    start = pd.Series(False, index=df.index)
    start[att.index] = ~prev.values
    # restarts: baseline is the average outcome of that restart type from that zone,
    # so the taker is credited only for beating the situation, not for being awarded it
    restart = df["action_type"].isin(RESTARTS)
    zone = [df["action_type"], df["action"], (df["x"] // 10), (df["y"].abs() // 10)]
    baseline = after_s.groupby(zone).transform("mean")
    won = start & ~restart
    df["xd_after"] = after
    df["xd_before"] = np.where(restart, baseline, before)
    df["xd_added"] = np.where(in_poss & ~won, df["xd_after"] - df["xd_before"], 0.0)
    df["xd_won"] = np.where(won, after, 0.0)
    # shots: the moment before a shot already prices the shot in, so taking one is not creation.
    # Its change in value is kept apart as shot selection and removed from xd_added.
    shot = (df["action_type"] == "SHOT").values
    df["xd_shot_sel"] = np.where(shot, df["xd_added"], 0.0)
    df["xd_added"] = np.where(shot, 0.0, df["xd_added"])
    # receptions: the value of controlling a completed delivery goes back to the passer
    nxt = df.shift(-1)
    hand = (df["action_type"].isin(DELIVERIES) & (df["result"] == "SUCCESS") & (nxt["action_type"] == "RECEPTION")
            & (nxt["squad_id"] == df["squad_id"]) & (nxt["match_id"] == df["match_id"])).values
    rec = np.flatnonzero(hand) + 1
    added = df["xd_added"].values.copy()
    # receiver value is kept as two separate metrics:
    #   xd_received: danger of the spot where the player gets the ball (movement, positioning)
    #   xd_rec_control: what the reception itself adds over the pass (first touch, control)
    received = np.zeros(len(df))
    control = np.zeros(len(df))
    received[rec] = df["xd_after"].values[rec]
    control[rec] = added[rec]
    df["xd_received"] = received
    df["xd_rec_control"] = control
    added[hand] += added[rec]
    added[rec] = 0.0
    df["xd_added"] = added
    # denied: the attacking team's latest value, ended by a successful defensive action
    mp = df["match_id"].astype(str) + "_" + df["period"].astype(str)
    last_att = after_s.where(in_poss).groupby(mp).ffill().groupby(mp).shift()
    last_sq = df["squad_id"].where(in_poss).groupby(mp).ffill().groupby(mp).shift()
    defensive = df["action_type"].isin(DEFENSIVE) & (df["result"] != "FAIL") & (last_sq != df["squad_id"])
    df["xd_denied"] = np.where(defensive, last_att.fillna(0), 0.0)
    return df



# ---------------------------------------------------------------- xG (02_xg.py)
def shot_features(df):
    s = df[df["action_type"] == "SHOT"].copy()
    s["abs_y"] = s["y"].abs()
    s["gk_goal_dist"] = np.hypot(52.5 - s["gk_x"], s["gk_y"])
    s["gk_shot_dist"] = np.hypot(s["gk_x"] - s["x"], s["gk_y"] - s["y"])
    s["set_piece"] = s["set_piece"].astype(int)
    s["goal"] = (s["result"] == "SUCCESS").astype(int)
    return s



# ---------------------------------------------------------------- xGOT (25_xgot.py)
XGOT_NUM = ["xg", "target_y", "abs_ty", "target_z", "post_gap", "bar_gap", "gk_x", "gk_y", "gk_ty_gap", "gk_ty_gap_flip",
       "shot_dist", "abs_y"]
XGOT_CAT = ["body_part", "action"]


def xgot_prep(d, cats):
    X = d[XGOT_NUM + XGOT_CAT].copy()
    for c in XGOT_CAT:
        X[c] = pd.Categorical(X[c], categories=cats[c])
    return X
