"""Danger events: dangerous moments that do not need a shot, valued by how often a goal follows."""
import numpy as np
import pandas as pd

from ..config import WINDOW_S

BOX_X, BOX_Y = 36.0, 20.16
GOLDEN_X, GOLDEN_Y = 41.5, 9.16  # in front of goal: penalty spot to goal line, six-yard box width

EVENTS = {
    "golden_zone_touch": "Touch between the penalty spot and goal, inside the six-yard box width",
    "cutback": "Completed pass from near the byline back into the box",
    "box_entry_pass": "Completed pass from outside into the box",
    "box_entry_carry": "Successful carry from outside into the box",
    "high_regain": "Interception or loose-ball regain in the final third",
    "corner": "Corner kick",
    "dangerous_free_kick": "Free kick within 30 m of goal",
}


def in_box(x, y):
    return (x >= BOX_X) & (np.abs(y) <= BOX_Y)


def detect(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean column per event type, for the acting team."""
    at, ok = df["action_type"], df["result"] == "SUCCESS"
    start_box, end_box = in_box(df.x, df.y), in_box(df.end_x, df.end_y)
    ev = pd.DataFrame(index=df.index)
    ev["golden_zone_touch"] = (df.in_poss == 1) & (df.x >= GOLDEN_X) & (df.y.abs() <= GOLDEN_Y)
    ev["cutback"] = (at == "PASS") & ok & (df.x >= 46.5) & (df.y.abs() >= 5.5) & end_box & (df.end_x < df.x - 1)
    ev["box_entry_pass"] = (at == "PASS") & ok & ~start_box & end_box
    ev["box_entry_carry"] = (at == "DRIBBLE") & ok & ~start_box & end_box
    ev["high_regain"] = at.isin(["LOOSE_BALL_REGAIN", "INTERCEPTION"]) & (df.result != "FAIL") & (df.x >= 17.5)
    ev["corner"] = at == "CORNER"
    ev["dangerous_free_kick"] = (at == "FREE_KICK") & (df.start_dist <= 30)
    return ev


def _blocks(df):
    keys = df["match_id"].astype(np.int64).values * 10 + df["period"].values
    sq, t = df["squad_id"].values, df["t"].values
    order = np.lexsort((t, sq, keys))
    k, s = keys[order], sq[order]
    block = np.r_[True, (k[1:] != k[:-1]) | (s[1:] != s[:-1])]
    comp = (np.cumsum(block) - 1) * 1e6 + t[order]
    return order, comp


def goal_follows(df: pd.DataFrame) -> np.ndarray:
    """1 if the acting team scores in [t, t + WINDOW_S] (same match period), else 0."""
    order, comp = _blocks(df)
    goal = ((df["action_type"] == "SHOT") & (df["result"] == "SUCCESS")).values[order].astype(float)
    cum0 = np.r_[0.0, np.cumsum(goal)]
    lo = np.searchsorted(comp, comp, side="left")
    hi = np.searchsorted(comp, comp + WINDOW_S, side="right")
    out = np.zeros(len(df))
    out[order] = (cum0[hi] - cum0[lo] > 0).astype(float)
    return out


def window_max(df: pd.DataFrame, value: np.ndarray) -> np.ndarray:
    """Max of value over the acting team's rows in (t, t + WINDOW_S], same match period."""
    order, comp = _blocks(df)
    v = value[order]
    n = len(v)
    lo = np.searchsorted(comp, comp, side="right")
    hi = np.searchsorted(comp, comp + WINDOW_S, side="right")
    # sparse table for range-max queries over [lo, hi)
    table = [v]
    k = 1
    while (1 << k) <= n:
        prev = table[-1]
        half = 1 << (k - 1)
        table.append(np.maximum(prev[:-half], prev[half:]) if len(prev) > half else prev[:0])
        k += 1
    length = hi - lo
    res = np.zeros(n)
    has = length > 0
    lvl = np.zeros(n, dtype=int)
    lvl[has] = np.floor(np.log2(length[has])).astype(int)
    for j in np.unique(lvl[has]):
        m = has & (lvl == j)
        tj = table[j]
        res[m] = np.maximum(tj[lo[m]], tj[hi[m] - (1 << j)])
    out = np.zeros(n)
    out[order] = res
    return out
