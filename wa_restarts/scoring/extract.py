"""IMPECT event JSON -> on-ball actions (xDanger 01_extract.py) and shot end data (23_extract_shot_targets.py)."""
import json
import sys

SKIP_TYPES = {"NO_VIDEO", "FINAL_WHISTLE", "YELLOW_CARD", "RED_CARD", "GOAL",
              "OFFSIDE", "SUBSTITUTION", "OUT"}
SAVES = {"GK_SAVE", "GK_CATCH"}

COLS = ["match_id", "date", "idx", "period", "t", "squad_id", "att_squad_id", "player_id",
        "player_name", "position", "action_type", "action", "phase", "result",
        "body_part", "prev_pass_height", "pressure", "opponents", "dist_to_opp",
        "x", "y", "end_x", "end_y", "pass_dist", "receiver_type", "set_piece",
        "sequence", "shot_dist", "shot_angle", "gk_x", "gk_y", "impect_pxt"]
TARGET_COLS = ["match_id", "idx", "target_y", "target_z", "woodwork", "dive_y", "dive_z", "gk_saved"]


def g(d, *keys):
    for k in keys:
        if d is None:
            return None
        d = d.get(k)
    return d


def parse_file(path):
    """(actions, shot targets) for one match file; None when the file cannot be read."""
    try:
        events = json.loads(path.read_bytes())
    except Exception as exc:
        print(f"skip {path.name}: {exc}", file=sys.stderr)
        return None
    match_id = int(path.stem.rsplit("_", 1)[-1])
    date = path.name[:10]
    rows = []
    for e in events:
        at = e.get("actionType")
        if at in SKIP_TYPES or e.get("player") is None:
            continue
        sx, sy = g(e, "start", "adjCoordinates", "x"), g(e, "start", "adjCoordinates", "y")
        if sx is None:
            continue
        shot = e.get("shot") or {}
        rows.append((
            match_id, date, e["index"], e["periodId"], g(e, "gameTime", "gameTimeInSec"),
            e["squadId"], e.get("currentAttackingSquadId"), g(e, "player", "id"),
            g(e, "player", "name"), g(e, "player", "position"),
            at, e.get("action"), e.get("phase"), e.get("result"),
            e.get("bodyPartExtended"), e.get("previousPassHeight"),
            e.get("pressure"), e.get("opponents"), e.get("distanceToOpponent"),
            sx, sy, g(e, "end", "adjCoordinates", "x"), g(e, "end", "adjCoordinates", "y"),
            g(e, "pass", "distance"), g(e, "pass", "receiver", "type"),
            e.get("setPiece") is not None or bool(e.get("inferredSetPiece")),
            e.get("sequenceIndex"),
            shot.get("distance"), shot.get("angle"),
            g(shot, "gk", "adjCoordinates", "x"), g(shot, "gk", "adjCoordinates", "y"),
            g(e, "pxT", "team"),
        ))
    return rows, targets(events, match_id)


def targets(ev, mid):
    rows = []
    for i, e in enumerate(ev):
        if e.get("actionType") != "SHOT" or not e.get("shot"):
            continue
        s = e["shot"]
        tp = s.get("targetPoint") or {}
        dive = (s.get("gk") or {}).get("divePoint") or {}
        # keeper save: the next few events contain a save or catch by the other team within 3 s
        t0 = e["gameTime"]["gameTimeInSec"]
        saved = False
        for nx in ev[i + 1:i + 6]:
            if nx["gameTime"]["gameTimeInSec"] - t0 > 3:
                break
            if nx.get("actionType") in SAVES and nx.get("squadId") != e.get("squadId"):
                saved = True
                break
        rows.append((mid, e["index"], tp.get("y"), tp.get("z"), s.get("woodwork"), dive.get("y"), dive.get("z"), saved))
    return rows
