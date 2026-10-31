"""Score one league-season from IMPECT event JSON with the saved models (no retraining).

Same steps as xDanger refresh_competition.py, but everything lands in cache/<league-season>/ and nothing in the xDanger
folder is touched: events, features, event labels, xDanger scored actions, xShot, shots (xG, targets, xGOT), restart
danger curves, restart possessions and the four-model restart windows.
"""
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import joblib
import numpy as np
import pandas as pd

from .. import config
from ..common import add_context, danger_label
from . import danger_events as de
from . import extract, features as fx, restarts


def log(msg):
    print(msg, flush=True)


def parse(files, workers=8):
    try:
        with ProcessPoolExecutor(workers) as pool:
            results = list(pool.map(extract.parse_file, files, chunksize=4))
    except BrokenProcessPool:  # workers cannot start (e.g. code piped through stdin): parse in this process
        log("parallel parsing unavailable, parsing in one process")
        results = [extract.parse_file(f) for f in files]
    rows = [r for res in results if res for r in res[0]]
    targets = [r for res in results if res for r in res[1]]
    return pd.DataFrame(rows, columns=extract.COLS), pd.DataFrame(targets, columns=extract.TARGET_COLS)


def run(comp, workers=8):
    ws = config.Workspace(comp)
    files = sorted(ws.json_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No IMPECT match files in {ws.json_dir}")
    ws.dir.mkdir(parents=True, exist_ok=True)

    # events (01) and shot end data (23)
    ev, tr = parse(files, workers)
    ev["competition"] = comp
    ev.to_parquet(ws.events, index=False)
    log(f"events: {len(files)} matches, {len(ev):,} actions")

    # features, xG and the danger label (03)
    xgm = joblib.load(config.model("xg_model"))
    df = add_context(ev.copy())
    df["set_piece"] = df["set_piece"].astype(int)
    df["xg"] = fx.xg_for(df, xgm)
    df["label"] = danger_label(df, df["xg"].values)
    feats = df[list(dict.fromkeys(fx.ID + fx.NUM + fx.CAT + ["xg", "label"]))]
    feats.to_parquet(ws.features, index=False)
    log(f"features: {len(feats):,} rows")

    # event labels: danger events valued by the training-season goal rates
    vals = pd.read_csv(config.table("danger_event_values.csv"))
    d = pd.read_parquet(ws.features, columns=["match_id", "period", "t", "squad_id", "action_type", "result",
                                              "in_poss", "x", "y", "end_x", "end_y", "start_dist", "xg"])
    det = de.detect(d)
    v = d["xg"].values.copy()
    for e, w in zip(vals.event, vals.value):
        v = np.where(det[e].values, np.maximum(v, w), v)
    top = det.idxmax(axis=1).where(det.any(axis=1))
    lab = pd.DataFrame({"label": de.window_max(d, v), "event": top.values})
    lab.to_parquet(ws.event_labels, index=False)

    # xDanger scored actions
    m = joblib.load(config.model("xdanger_events_model"))
    df = pd.read_parquet(ws.features)
    df["label"], df["danger_event"] = lab["label"].values, lab["event"].values
    df["xdanger"] = m["model"].predict(fx.prep(df, m["cats"]))
    shot = (df["action_type"] == "SHOT").values
    df = fx.credit(df, np.where(shot, np.maximum(df["xg"], df["xdanger"]), df["xdanger"]))
    keep = fx.ID + ["in_poss", "xg", "label", "xdanger", "xd_after", "xd_before", "xd_added", "xd_won", "xd_denied",
                    "xd_shot_sel", "xd_received", "xd_rec_control", "danger_event"]
    scored = df[keep]
    scored.to_parquet(ws.scored, index=False)
    log("xDanger scored")

    # shots: xG (02), targets (23), xGOT (25)
    s = fx.shot_features(add_context(ev.copy()))
    for c in xgm["cat"]:
        s[c] = pd.Categorical(s[c], categories=xgm["cats"][c])
    s["xg"] = xgm["model"].predict_proba(s[xgm["num"] + xgm["cat"]])[:, 1]
    s = s[["competition", "match_id", "idx", "squad_id", "player_id", "player_name", "x", "y", "action", "body_part",
           "set_piece", "goal", "xg", "gk_x", "gk_y", "shot_dist"]]
    s = s.merge(tr, on=["match_id", "idx"], how="left")
    for c in ["gk_x", "gk_y", "shot_dist", "target_y", "target_z", "dive_y", "dive_z"]:
        s[c] = pd.to_numeric(s[c], errors="coerce")
    in_frame = (s.target_y.abs() <= 3.66) & (s.target_z <= 2.44) & s.woodwork.isna()
    s["on_target"] = (s.goal == 1) | (in_frame & s.gk_saved.fillna(False).astype(bool))
    s["abs_ty"] = s.target_y.abs()
    s["post_gap"] = 3.66 - s.abs_ty
    s["bar_gap"] = 2.44 - s.target_z
    s["gk_ty_gap"] = (s.target_y - s.gk_y).abs()
    s["gk_ty_gap_flip"] = (s.target_y + s.gk_y).abs()
    s["abs_y"] = s.y.abs()
    gm = joblib.load(config.model("xgot_model"))
    s["xgot"] = 0.0
    if s.on_target.any():
        s.loc[s.on_target, "xgot"] = gm["model"].predict_proba(fx.xgot_prep(s[s.on_target], gm["cats"]))[:, 1]
    shots = s[["competition", "match_id", "idx", "squad_id", "player_id", "player_name", "x", "y", "action", "body_part",
               "set_piece", "goal", "xg", "xgot", "on_target", "target_y", "target_z", "gk_x", "gk_y"]]
    shots.to_parquet(ws.shots, index=False)
    log(f"shots: {len(shots):,} (xG {shots.xg.sum():.1f}, goals {int(shots.goal.sum())})")

    # xShot (24)
    sm = joblib.load(config.model("xshot_model"))
    feats = pd.read_parquet(ws.features)
    feats = feats.loc[:, ~feats.columns.duplicated()]
    xs = sm["model"].predict_proba(fx.prep(feats, sm["cats"]))[:, 1]
    c = fx.credit(feats.copy(), np.where((feats.action_type == "SHOT").values, 1.0, xs))
    xshot = pd.DataFrame({"match_id": feats.match_id.values, "idx": feats.idx.values, "xshot": xs,
                          "xs_added": c.xd_added.values, "xs_won": c.xd_won.values})
    xshot.to_parquet(ws.xshot, index=False)
    log("xShot scored")

    # restarts: danger curves (14, 15), possessions (20), four-model windows (31)
    cur = restarts.score_curves(restarts.curves(scored, comp))
    cur.to_parquet(ws.curves, index=False)
    pos = restarts.possessions(scored, comp)
    pos.to_parquet(ws.possessions, index=False)
    fm = restarts.four_models(scored, xshot, shots, comp)
    fm.to_parquet(ws.four_models, index=False)
    log(f"restarts: {len(fm):,} windows, {len(cur):,} curves")
    return ws
