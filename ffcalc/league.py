"""Turn raw Sleeper league JSON into the League model."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ffcalc.clients.sleeper import check_league_id, check_username
from ffcalc.models import NON_STARTING, League, Team


class LeagueNotFound(ValueError):
    def __init__(self, league_id: str):
        super().__init__(
            f"Sleeper league {league_id!r} not found. A league ID is a long number, "
            "like 1400221038771437568 (see the league's URL in the Sleeper app)."
        )
        self.league_id = league_id


def _pairs(rows: list[dict]) -> list[tuple[int, int]]:
    by_id: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        if r.get("matchup_id") is not None:
            by_id[r["matchup_id"]].append(r["roster_id"])
    return [(a[0], a[1]) for a in by_id.values() if len(a) == 2]


def _team(roster: dict, users: dict[str, dict]) -> Team:
    s = roster.get("settings") or {}
    reserve = list(roster.get("reserve") or [])
    players = list(dict.fromkeys([*(roster.get("players") or []), *reserve]))
    user = users.get(roster.get("owner_id") or "", {})
    name = (
        (user.get("metadata") or {}).get("team_name")
        or user.get("display_name")
        or f"Team {roster['roster_id']}"
    )
    return Team(
        roster_id=roster["roster_id"],
        owner_id=roster.get("owner_id"),
        name=name,
        players=players,
        reserve=reserve,
        wins=s.get("wins", 0),
        losses=s.get("losses", 0),
        ties=s.get("ties", 0),
        points_for=s.get("fpts", 0) + s.get("fpts_decimal", 0) / 100,
        points_against=s.get("fpts_against", 0) + s.get("fpts_against_decimal", 0) / 100,
    )


def load_league(league_id: str, sleeper: Any) -> League:
    league_id = check_league_id(league_id.strip())
    raw = sleeper.league(league_id)
    if not raw:
        raise LeagueNotFound(league_id)
    st = raw.get("settings") or {}
    positions = raw.get("roster_positions") or []
    start_week = st.get("start_week") or 1
    playoff_start = st.get("playoff_week_start") or 15
    current_week = st["last_scored_leg"] + 1 if "last_scored_leg" in st else start_week

    users = {u["user_id"]: u for u in sleeper.users(league_id)}
    teams = [_team(r, users) for r in sleeper.rosters(league_id)]

    schedule: dict[int, list[tuple[int, int]]] = {}
    results: dict[int, dict[int, float]] = {}
    for week in range(start_week, playoff_start):
        rows = sleeper.matchups(league_id, week)
        schedule[week] = _pairs(rows)
        if week < current_week:
            results[week] = {r["roster_id"]: float(r.get("points") or 0.0) for r in rows}

    prev = raw.get("previous_league_id")
    return League(
        league_id=str(raw.get("league_id", league_id)),
        name=raw.get("name", "").strip(),
        season=int(raw["season"]),
        num_teams=st.get("num_teams") or raw.get("total_rosters") or len(teams),
        slots=[p for p in positions if p not in NON_STARTING],
        bench_size=positions.count("BN"),
        ir_slots=st.get("reserve_slots", 0),
        scoring={k: float(v) for k, v in (raw.get("scoring_settings") or {}).items()},
        playoff_teams=st.get("playoff_teams", 0),
        playoff_week_start=playoff_start,
        playoff_round_type=st.get("playoff_round_type", 0),
        median_win=st.get("league_average_match") == 1,
        trade_deadline=st.get("trade_deadline", 0),
        current_week=current_week,
        teams=teams,
        schedule=schedule,
        results=results,
        previous_league_id=None if prev in (None, "", "0") else str(prev),
    )


def find_leagues(username: str, season: int, sleeper: Any) -> list[dict]:
    user = sleeper.user(check_username(username.strip()))
    if not user:
        return []
    return [
        {"league_id": lg["league_id"], "name": lg["name"], "total_rosters": lg["total_rosters"]}
        for lg in sleeper.user_leagues(user["user_id"], season) or []
    ]
