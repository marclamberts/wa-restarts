"""Feature engineering shared by the scoring steps (same as xDanger/scripts/common.py)."""
import numpy as np
import pandas as pd

from .config import TEST_SEASONS, WINDOW_S

GOAL_X, POST_Y = 52.5, 3.66
DIST_OPP = {"LESS_THAN_ONE_METER": 0.5, "ONE_METER": 1, "TWO_METERS": 2,
            "THREE_METERS": 3, "FOUR_METERS": 4, "MORE_THAN_FOUR_METERS": 6}


def is_test(competition: pd.Series) -> pd.Series:
    return competition.str.endswith(TEST_SEASONS[0]) | competition.str.endswith(" " + TEST_SEASONS[1])


def goal_geometry(x, y):
    dx = GOAL_X - x
    dist = np.hypot(dx, y)
    a1 = np.arctan2(POST_Y - y, dx)
    a2 = np.arctan2(-POST_Y - y, dx)
    mouth = np.degrees(np.abs(a1 - a2))
    return dist, mouth


def load(path, columns=None, filters=None):
    return pd.read_parquet(path, columns=columns, filters=filters)


def add_context(df: pd.DataFrame) -> pd.DataFrame:
    """Location, previous-action and possession context. df must be sorted by match_id, idx."""
    df = df.sort_values(["match_id", "idx"]).reset_index(drop=True)
    num = ["t", "pressure", "opponents", "x", "y", "end_x", "end_y", "pass_dist",
           "shot_dist", "shot_angle", "gk_x", "gk_y", "impect_pxt"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce").astype(float)
    df["end_x"] = df["end_x"].fillna(df["x"])
    df["end_y"] = df["end_y"].fillna(df["y"])
    df["dist_opp_m"] = df["dist_to_opp"].map(DIST_OPP)
    df["in_poss"] = (df["squad_id"] == df["att_squad_id"]).astype(np.int8)
    df["start_dist"], df["start_mouth"] = goal_geometry(df["x"].values, df["y"].values)
    df["end_dist"], df["end_mouth"] = goal_geometry(df["end_x"].values, df["end_y"].values)
    df["abs_end_y"] = df["end_y"].abs()
    df["dx"] = df["end_x"] - df["x"]

    same_mp = (df["match_id"] == df["match_id"].shift()) & (df["period"] == df["period"].shift())
    for k in (1, 2):
        grp = same_mp if k == 1 else same_mp & same_mp.shift(fill_value=False)
        same_team = grp & (df["squad_id"] == df["squad_id"].shift(k))
        sign = np.where(same_team, 1.0, -1.0)  # flip coordinates of opponent actions into actor frame
        df[f"p{k}_type"] = df["action"].shift(k).where(grp)
        df[f"p{k}_same"] = np.where(grp, same_team.astype(float), np.nan)
        df[f"p{k}_end_x"] = (df["end_x"].shift(k) * sign).where(grp)
        df[f"p{k}_end_y"] = (df["end_y"].shift(k) * sign).where(grp)
        df[f"p{k}_dt"] = (df["t"] - df["t"].shift(k)).where(grp)

    # possession = run of rows with the same attacking squad inside a match period
    new_poss = ~same_mp | (df["att_squad_id"] != df["att_squad_id"].shift())
    poss = new_poss.cumsum()
    first = df.groupby(poss)[["t", "x", "squad_id"]].transform("first")
    df["poss_t"] = df["t"] - first["t"]
    df["poss_n"] = df.groupby(poss).cumcount()
    df["poss_start_x"] = np.where(first["squad_id"] == df["squad_id"], first["x"], -first["x"])
    df["poss_id"] = poss.values
    return df


def danger_label(df: pd.DataFrame, xg: np.ndarray) -> np.ndarray:
    """P(goal) from shots by the actor's team in (t, t+WINDOW_S], same match period."""
    out = np.zeros(len(df))
    is_shot = (df["action_type"] == "SHOT").values
    log1m = np.log1p(-np.clip(xg, 0, 0.999))
    keys = df["match_id"].astype(np.int64).values * 10 + df["period"].values
    t = df["t"].values
    sq = df["squad_id"].values
    order = np.lexsort((t, sq, keys))
    k_s, s_s, t_s = keys[order], sq[order], t[order]
    shot_s = is_shot[order]
    # cumulative log(1-xG) over shots within each (match, period, squad) block
    cum0 = np.r_[0.0, np.cumsum(np.where(shot_s, log1m[order], 0.0))]
    block = np.r_[True, (k_s[1:] != k_s[:-1]) | (s_s[1:] != s_s[:-1])]
    # composite sortable key: block index spaced far apart, then time
    comp = (np.cumsum(block) - 1) * 1e6 + t_s
    lo = np.searchsorted(comp, comp, side="right")  # strictly after t
    hi = np.searchsorted(comp, comp + WINDOW_S, side="right")
    s = cum0[hi] - cum0[lo]
    out[order] = -np.expm1(s)
    return out
