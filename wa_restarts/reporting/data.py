"""League-season data for the reports: actions with xDanger, shots with xGOT, classified restarts, the match list
(scores, matchday, formations from IMPECT meta when present) and per team-match and player-match metrics, which give
every report its league context (percentiles)."""
import difflib
import json
from functools import lru_cache

import numpy as np
import pandas as pd

from .. import config
from .. import options as ro
from .. import xdanger_tables as xt
from ..names import home_squads

POS_ABBR = {"GOALKEEPER": "GK", "CENTRAL_DEFENDER": "CB", "RIGHT_WINGBACK_DEFENDER": "RB", "LEFT_WINGBACK_DEFENDER": "LB",
            "DEFENSE_MIDFIELD": "DM", "CENTRAL_MIDFIELD": "CM", "ATTACKING_MIDFIELD": "AM", "RIGHT_WINGER": "RW",
            "LEFT_WINGER": "LW", "CENTER_FORWARD": "CF"}
POS_GROUP = {"GK": "Goalkeepers", "CB": "Centre backs", "RB": "Full backs", "LB": "Full backs", "DM": "Midfielders",
             "CM": "Midfielders", "AM": "Attacking midfielders", "RW": "Wingers", "LW": "Wingers", "CF": "Forwards"}
DEFENSIVE = ["INTERCEPTION", "CLEARANCE", "BLOCK", "GROUND_DUEL", "LOOSE_BALL_REGAIN"]
ATTACKING_RESTARTS = ["Corner", "Short corner", "Indirect free kick", "Direct free kick", "Penalty", "Throw-in"]

# team-match metrics: key -> (label, format, higher is better)
TEAM_METRICS = {
    "goals": ("Goals", "{:.0f}", True), "xg": ("xG", "{:.2f}", True), "shots": ("Shots", "{:.0f}", True),
    "on_target": ("Shots on target", "{:.0f}", True), "xgot": ("xGOT", "{:.2f}", True),
    "xg_per_shot": ("xG per shot", "{:.2f}", True), "xdanger": ("xDanger created", "{:.2f}", True),
    "possession": ("Possession %", "{:.0%}", True), "passes": ("Passes", "{:.0f}", True),
    "pass_pct": ("Pass completion %", "{:.0%}", True), "progressive": ("Progressive passes", "{:.0f}", True),
    "final_third": ("Final-third entries", "{:.0f}", True), "box_entries": ("Box entries", "{:.0f}", True),
    "crosses": ("Crosses", "{:.0f}", True), "field_tilt": ("Field tilt %", "{:.0%}", True),
    "regains": ("Ball regains", "{:.0f}", True), "high_regains": ("Final-third regains", "{:.0f}", True),
    "def_actions": ("Defensive actions", "{:.0f}", True), "ppda": ("PPDA (lower = more pressing)", "{:.1f}", False),
    "sp_xg": ("Set-piece xG", "{:.2f}", True), "corners": ("Corners", "{:.0f}", True), "fouls": ("Fouls", "{:.0f}", False),
}
PLAYER_METRICS = {
    "touches": ("Touches", "{:.0f}"), "passes": ("Passes", "{:.0f}"), "pass_pct": ("Pass completion %", "{:.0%}"),
    "progressive": ("Progressive passes", "{:.0f}"), "prog_carries": ("Progressive carries", "{:.0f}"),
    "box_entries": ("Box entries", "{:.0f}"), "shots": ("Shots", "{:.0f}"), "xg": ("xG", "{:.2f}"),
    "xd_added": ("xDanger added", "{:+.2f}"), "xd_received": ("xDanger received", "{:.2f}"),
    "regains": ("Ball regains", "{:.0f}"), "def_actions": ("Defensive actions", "{:.0f}"), "xd_denied": ("xDanger denied", "{:.2f}"),
}


def minute(period, t):
    period = np.asarray(period)
    t = np.asarray(t, dtype=float)
    return np.select([period == 1, period == 2, period == 3], [t / 60, 45 + (t - 10000) / 60, 90 + (t - 20000) / 60],
                     105 + (t - 30000) / 60)


def _progressive(d):
    sd = np.hypot(52.5 - d.x, d.y)
    ed = np.hypot(52.5 - d.end_x, d.end_y)
    return (d.result == "SUCCESS") & ((sd - ed) >= np.maximum(10, 0.25 * sd)) & (d.end_x > -17.5)


class League:
    def __init__(self, comp, refresh=False):
        self.comp = comp
        a = xt.actions(comp, refresh=refresh)
        a["minute"] = minute(a.period, a.t)
        a["pos"] = a.position.map(POS_ABBR).fillna("-")
        self.actions = a
        ws = config.Workspace(comp)
        names = a.drop_duplicates("squad_id").set_index("squad_id").team
        sh = pd.read_parquet(ws.shots).merge(a[["match_id", "idx", "period", "t", "minute", "team", "result"]],
                                            on=["match_id", "idx"], how="left")
        sh["on_target"] = sh.on_target.fillna(False).astype(bool)
        sh["penalty"] = sh.action == "PENALTY_KICK"
        self.shots = sh
        r = ro.classify(ro.load(comp))
        r["team"] = r.squad_id.map(names)
        r = r.merge(a[["match_id", "idx", "period", "t", "minute", "player_name"]], on=["match_id", "idx"], how="left")
        r["attacking"] = r.restart.isin(ATTACKING_RESTARTS) & ~((r.restart == "Indirect free kick") & r.option.str.startswith("Own half")) \
            & ~((r.restart == "Throw-in") & r.option.str.startswith("Elsewhere"))
        self.restarts = r
        self.matches = self._matches(names)
        r["opponent"] = [self.opponent(m, t) for m, t in zip(r.match_id, r.team)]
        self.team_match = self._team_match()
        self.player_match = self._player_match()

    # ------------------------------------------------------------------ matches
    def _matches(self, names):
        a = self.actions
        meta = {}
        f = config.Workspace(self.comp).meta_dir / "matches.json"
        if f.exists():
            meta = {m["id"]: m for m in json.loads(f.read_text())}
        lineups = {}
        lf = config.Workspace(self.comp).meta_dir / "lineups.json"
        if lf.exists():
            lineups = {int(k): v for k, v in json.loads(lf.read_text()).items()}
        homes = home_squads(self.comp)
        rows = []
        for mid, g in a.groupby("match_id"):
            sq = list(dict.fromkeys(g.squad_id))
            if len(sq) < 2:
                continue
            h = homes.get(mid, sq[0])
            aw = next(s for s in sq if s != h)
            m = meta.get(mid, {})
            goals = m.get("goals") or {}
            if goals.get("home", {}).get("fullTime") is not None:
                hg, ag = goals["home"]["fullTime"], goals["away"]["fullTime"]
                hht, aht = goals["home"].get("halfTime"), goals["away"].get("halfTime")
            else:  # from events: goals scored plus own goals by the opponent
                sc = lambda s, o: int(((g.squad_id == s) & (g.action_type == "SHOT") & (g.result == "SUCCESS")).sum()
                                      + ((g.squad_id == o) & (g.action_type == "OWN_GOAL")).sum())
                hg, ag, hht, aht = sc(h, aw), sc(aw, h), None, None
            lu = lineups.get(mid, {})
            rows.append({"match_id": mid, "date": g.date.iloc[0], "home_id": h, "away_id": aw, "home": names.get(h, str(h)),
                         "away": names.get(aw, str(aw)), "home_goals": hg, "away_goals": ag, "home_ht": hht, "away_ht": aht,
                         "matchday": (m.get("matchDay") or {}).get("name"),
                         "home_formation": (lu.get("home") or {}).get("formation"),
                         "away_formation": (lu.get("away") or {}).get("formation")})
        return pd.DataFrame(rows).sort_values(["date", "match_id"]).reset_index(drop=True)

    def opponent(self, mid, team):
        m = self._mrow(mid)
        return m.away if m.home == team else m.home

    def _mrow(self, mid):
        if not hasattr(self, "_mindex"):
            self._mindex = self.matches.set_index("match_id")
        return self._mindex.loc[mid]

    def teams(self):
        return sorted(set(self.matches.home) | set(self.matches.away))

    def team(self, name):
        names = self.teams()
        if name in names:
            return name
        hit = [n for n in names if name.lower() in n.lower()]
        if len(hit) == 1:
            return hit[0]
        raise ValueError(f"Team '{name}' not in {self.comp}. Did you mean: {hit or difflib.get_close_matches(name, names, 5, 0.3)}")

    def match(self, match=None, team=None):
        """A match row: match id, 'Home vs Away' (partial names work), or a team's latest (match=None or 'latest')."""
        m = self.matches
        if isinstance(match, (int, np.integer)) or (isinstance(match, str) and match.isdigit()):
            hit = m[m.match_id == int(match)]
        elif isinstance(match, str) and " vs " in match.lower():
            h, a_ = [s.strip() for s in match.lower().split(" vs ")]
            hit = m[m.home.str.lower().str.contains(h, regex=False) & m.away.str.lower().str.contains(a_, regex=False)]
        else:
            if team is None:
                raise ValueError("give a match ('Home vs Away' or a match id) or a team for its latest match")
            t = self.team(team)
            hit = m[(m.home == t) | (m.away == t)]
            if isinstance(match, str) and match not in ("latest", ""):
                hit = hit[hit.date == match] if match[:4].isdigit() else hit[(hit.home + " " + hit.away).str.lower().str.contains(match.lower())]
            hit = hit.tail(1)
        if hit.empty:
            raise ValueError(f"No match found for {match!r} (team {team!r}) in {self.comp}")
        if len(hit) > 1:
            hit = hit.tail(1)
        return hit.iloc[0]

    def player(self, name, team=None):
        a = self.actions if team is None else self.actions[self.actions.team == self.team(team)]
        names = a.drop_duplicates("player_id")[["player_id", "player_name", "team"]]
        hit = names[names.player_name == name]
        if hit.empty:
            hit = names[names.player_name.str.lower().str.contains(name.lower(), regex=False, na=False)]
        if len(hit) == 1:
            return hit.iloc[0]
        opts = difflib.get_close_matches(name, names.player_name.dropna().tolist(), 6, 0.4) if hit.empty else hit.player_name.tolist()
        raise ValueError(f"Player '{name}' {'is ambiguous' if len(hit) else 'not found'} in {self.comp}: {opts}")

    # ------------------------------------------------------------------ metrics
    def _team_match(self):
        a, rows = self.actions, []
        sh = self.shots
        att = self.restarts[self.restarts.attacking]
        for (mid, team), g in a.groupby(["match_id", "team"]):
            opp_team = self.opponent(mid, team)
            o = a[(a.match_id == mid) & (a.team == opp_team)]
            s = sh[(sh.match_id == mid) & (sh.team == team)]
            passes = g[g.action_type == "PASS"]
            opp_passes = o[o.action_type == "PASS"]
            ip = g[g.in_poss == 1]
            ft_own = (ip.x > 17.5).sum()
            ft_opp = ((o.in_poss == 1) & (o.x > 17.5)).sum()
            moves = ip[ip.action_type.isin(["PASS", "DRIBBLE"]) & (ip.result == "SUCCESS")]
            press = g[g.action_type.isin(["INTERCEPTION", "GROUND_DUEL", "FOUL", "LOOSE_BALL_REGAIN", "BLOCK"]) & (g.x > -10.5)]
            mr = self._mrow(mid)
            goals = mr.home_goals if mr.home == team else mr.away_goals
            rows.append({
                "match_id": mid, "team": team, "opponent": opp_team, "home": mr.home == team, "goals": goals,
                "xg": s.xg.sum(), "shots": len(s), "on_target": int(s.on_target.sum()), "xgot": s.xgot.sum(),
                "xg_per_shot": s.xg.mean() if len(s) else np.nan,
                "xdanger": np.where(g.in_poss == 1, g.xd_added.clip(lower=0), 0).sum(),
                "passes": len(passes), "pass_pct": (passes.result == "SUCCESS").mean() if len(passes) else np.nan,
                "possession": len(passes) / max(len(passes) + len(opp_passes), 1),
                "progressive": int(_progressive(passes).sum()),
                "final_third": int(((moves.x <= 17.5) & (moves.end_x > 17.5)).sum()),
                "box_entries": int(g.danger_event.isin(["box_entry_pass", "box_entry_carry"]).sum()),
                "crosses": int(g.action.isin(["HIGH_CROSS", "LOW_CROSS"]).sum()),
                "field_tilt": ft_own / max(ft_own + ft_opp, 1),
                "regains": int((g.action_type.isin(["LOOSE_BALL_REGAIN", "INTERCEPTION"]) & (g.result != "FAIL")).sum()),
                "high_regains": int((g.danger_event == "high_regain").sum()),
                "def_actions": int(g.action_type.isin(DEFENSIVE).sum()),
                "ppda": ((opp_passes.x < 10.5).sum() / max(len(press), 1)),
                "sp_xg": att[(att.match_id == mid) & (att.team == team)].xg_for.sum(),
                "corners": int(((self.restarts.match_id == mid) & (self.restarts.team == team)
                                & self.restarts.restart.isin(["Corner", "Short corner"])).sum()),
                "fouls": int((g.action_type == "FOUL").sum())})
        t = pd.DataFrame(rows)
        opp = t.set_index(["match_id", "team"])
        for k in ["goals", "xg", "shots", "xdanger", "sp_xg"]:
            t[f"{k}_against"] = [opp.loc[(m, o), k] for m, o in zip(t.match_id, t.opponent)]
        return t

    def _player_match(self):
        a = self.actions
        mins = xt.minutes(a).rename("minutes")
        rows = []
        for (mid, pid), g in a.groupby(["match_id", "player_id"]):
            passes = g[g.action_type == "PASS"]
            carries = g[g.action_type == "DRIBBLE"]
            rows.append({"match_id": mid, "player_id": pid, "player": g.player_name.iloc[0], "team": g.team.iloc[0],
                         "pos": g.pos.mode().iat[0] if g.pos.notna().any() else "-",
                         "touches": int((g.action_type != "RECEPTION").sum() + (g.action_type == "RECEPTION").sum()),
                         "passes": len(passes), "pass_pct": (passes.result == "SUCCESS").mean() if len(passes) else np.nan,
                         "progressive": int(_progressive(passes).sum()), "prog_carries": int(_progressive(carries).sum()),
                         "box_entries": int(g.danger_event.isin(["box_entry_pass", "box_entry_carry"]).sum()),
                         "shots": int((g.action_type == "SHOT").sum()), "xg": g.xg.sum(), "xd_added": g.xd_added.sum(),
                         "xd_received": g.xd_received.sum(), "xd_denied": g.xd_denied.sum(),
                         "regains": int((g.action_type.isin(["LOOSE_BALL_REGAIN", "INTERCEPTION"]) & (g.result != "FAIL")).sum()),
                         "def_actions": int(g.action_type.isin(DEFENSIVE).sum()),
                         "goals": int(((g.action_type == "SHOT") & (g.result == "SUCCESS")).sum())})
        p = pd.DataFrame(rows)
        p = p.join(mins, on=["match_id", "player_id"])
        p["group"] = p.pos.map(POS_GROUP).fillna("Other")
        return p


@lru_cache(maxsize=4)
def league(comp, style_key=None):
    """Cached League per competition (data does not depend on the style)."""
    return League(comp)


def pct_rank(value, pool, higher_better=True):
    pool = pd.Series(pool).dropna()
    if pool.empty or value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    p = (pool < value).mean() + 0.5 * (pool == value).mean()
    return p if higher_better else 1 - p
