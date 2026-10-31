"""Command line: wa-restarts (or python -m wa_restarts).

    wa-restarts leagues
    wa-restarts teams "2. Bundesliga 2026-2027"
    wa-restarts league "2. Bundesliga 2026-2027" --focus "Hertha BSC"
    wa-restarts club "2. Bundesliga 2026-2027" "Hertha" "St. Pauli"
    wa-restarts build "2. Bundesliga 2026-2027" --club "Hertha BSC"     # league folder + club packs
"""
import argparse

import matplotlib

matplotlib.use("Agg")

import wa_restarts as wr  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wa-restarts", description="IMPECT event data -> WA set-piece and restart deliverables")
    ap.add_argument("--json-root", help="folder with one sub-folder of IMPECT match JSON per league-season (default: ./events_json)")
    ap.add_argument("--meta-root", help="folder with <league-season>/squads.json, matches.json (default: events_meta next to the JSON root)")
    ap.add_argument("--out", help="deliverables root (default: ./wa_restarts_output)")
    ap.add_argument("--cache", help="where scored parquet files are kept (default: ~/.cache/wa_restarts)")
    ap.add_argument("--author", help="name in the chart header and footer")
    ap.add_argument("--brand", help="wordmark in the chart header")
    ap.add_argument("--mark", help="short mark in the chart header, e.g. WA")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("leagues", help="league-seasons with event JSON")
    p = sub.add_parser("teams", help="clubs in a league-season")
    p.add_argument("league")
    p = sub.add_parser("score", help="score a league-season into the cache only")
    p.add_argument("league")
    p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("league", help="league folder: O_* charts, B* visual pack and CSVs")
    p.add_argument("league")
    p.add_argument("--focus", help="club that also gets its own pizza in the visual pack")
    p.add_argument("--refresh", action="store_true", help="rescore from the JSON even if cached")
    p = sub.add_parser("club", help="set-piece pack for one or more clubs")
    p.add_argument("league")
    p.add_argument("teams", nargs="+")
    p.add_argument("--refresh", action="store_true")
    p = sub.add_parser("build", help="league folder plus club packs")
    p.add_argument("league")
    p.add_argument("--club", action="append", default=[], help="repeat for several clubs")
    p.add_argument("--refresh", action="store_true")
    a = ap.parse_args(argv)
    wr.configure(**{k: v for k, v in {"json_root": a.json_root, "meta_root": a.meta_root, "out_root": a.out,
                                      "cache": a.cache, "author": a.author, "brand": a.brand,
                                      "mark": a.mark}.items() if v})
    if a.cmd == "leagues":
        print("\n".join(wr.leagues()))
    elif a.cmd == "teams":
        print("\n".join(wr.teams(a.league)))
    elif a.cmd == "score":
        wr.score(a.league, refresh=True, workers=a.workers)
    elif a.cmd == "league":
        wr.league(a.league, focus_team=a.focus, refresh=a.refresh)
    elif a.cmd == "club":
        for i, t in enumerate(a.teams):
            wr.club(a.league, t, refresh=a.refresh and i == 0)
    elif a.cmd == "build":
        wr.build(a.league, teams=a.club, refresh=a.refresh)


if __name__ == "__main__":
    main()
