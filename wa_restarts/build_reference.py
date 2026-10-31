"""Copy what the package needs from the xDanger project into reference/ (run once, or after retraining).

    python -m wa_restarts.build_reference /path/to/xDanger

- models/: xG, xDanger (events), xShot and xGOT models
- danger_event_values.csv: goal rate after each danger event in the training seasons (event labels)
- restart_options_all_leagues.csv: every restart option across all leagues (the "all leagues" comparison columns)
- curve_expectations.parquet / curve_expectations_type.parquet: mean danger curves and peaks per restart type and
  10 m zone from the training seasons (what 15_restart_curve_model.py compares each restart with)
"""
import shutil
from pathlib import Path

import pandas as pd

from . import config
from .common import is_test

MODELS = ["xg_model", "xdanger_events_model", "xshot_model", "xgot_model"]
F = [f"f{i}" for i in range(31)]
A = [f"a{i}" for i in range(31)]
KEY = ["restart_type", "xb", "yb"]


def main():
    import sys
    src = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else config.XDANGER
    if src is None:
        raise SystemExit("usage: python -m wa_restarts.build_reference /path/to/xDanger (or set WA_RESTARTS_XDANGER)")
    out = config.REFERENCE
    (out / "models").mkdir(parents=True, exist_ok=True)
    for m in MODELS:
        shutil.copy2(src / "models" / f"{m}.joblib", out / "models" / f"{m}.joblib")
    shutil.copy2(src / "outputs" / "danger_event_values.csv", out / "danger_event_values.csv")
    shutil.copy2(src / "outputs" / "restarts" / "options" / "restart_options.csv", out / "restart_options_all_leagues.csv")

    zone, typ = [], []
    for f in sorted((src / "data" / "restarts" / "curves").glob("*.parquet")):
        if is_test(pd.Series([f.stem])).iloc[0]:
            continue
        d = pd.read_parquet(f, columns=["restart_type", "x", "y", "peak_for"] + F + A)
        d[F + A + ["peak_for"]] = d[F + A + ["peak_for"]].astype("float64")
        d["xb"] = (d.x.clip(-52.5, 52.4) // 10).astype(int)
        d["yb"] = (d.y.abs().clip(0, 33.9) // 10).astype(int)
        cols = F + A + ["peak_for"]
        zone.append(d.groupby(KEY)[cols].sum().assign(n=d.groupby(KEY).size()))
        typ.append(d.groupby("restart_type")[cols].sum().assign(n=d.groupby("restart_type").size()))
        print(f.stem, len(d), flush=True)
    z = pd.concat(zone).groupby(level=[0, 1, 2]).sum()
    t = pd.concat(typ).groupby(level=0).sum()
    cols = F + A + ["peak_for"]
    z[cols] = z[cols].div(z.n, axis=0)
    t[cols] = t[cols].div(t.n, axis=0)
    z.reset_index().to_parquet(out / "curve_expectations.parquet", index=False)
    t.reset_index().to_parquet(out / "curve_expectations_type.parquet", index=False)
    print(f"reference written to {out}")


if __name__ == "__main__":
    main()
