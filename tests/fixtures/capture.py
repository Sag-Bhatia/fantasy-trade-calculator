"""Refresh tests/fixtures/{sleeper,fantasycalc} from the live test league.

Run by hand (`uv run python tests/fixtures/capture.py`); tests never call this.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ffcalc import cache  # noqa: E402
from ffcalc.clients.espn import STAT_MAP, EspnClient  # noqa: E402
from ffcalc.clients.sleeper import (  # noqa: E402
    API,
    FANTASY_POSITIONS,
    PROJ_API,
    SleeperClient,
    _pos_params,
)

LEAGUE = "1400221038771437568"
SEASON = 2026
OUT = Path(__file__).parent
TOP_N = 40
# Row fields no code reads; dropping them keeps fixtures small.
_JUNK = ("last_modified", "updated_at", "week_shard", "date", "sport", "season_type")


def dump(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":")))
    print(f"{path.relative_to(OUT)}: {path.stat().st_size / 1e3:.0f} KB")


def slim(rows: list[dict], keep: set[str]) -> list[dict]:
    out = []
    for r in rows:
        if r["player_id"] not in keep:
            continue
        r = {k: v for k, v in r.items() if k not in _JUNK}
        r["player"] = {k: v for k, v in r["player"].items() if k != "metadata"}
        out.append(r)
    return out


def trim_espn(raw: dict, season: int, per_position: int) -> dict:
    """Keep the first few players per position (the response is sorted by ownership) and
    only weekly lines of `season`, plus one other-season and one season-level line on the first
    player so the client's filters are exercised. Stat dicts keep mapped ids + 2 unknown ids."""
    seen: dict[int, int] = {}
    players = []
    for e in raw["players"]:
        p = e["player"]
        pos = p["defaultPositionId"]
        if pos not in (1, 2, 3, 4) or seen.get(pos, 0) >= per_position:
            continue
        seen[pos] = seen.get(pos, 0) + 1
        stats, extras = [], 0
        for s in p["stats"]:
            weekly = s["statSplitTypeId"] == 1
            keep = weekly and s["seasonId"] == season
            if not keep and not players and extras < 2 and (weekly or s["seasonId"] == season):
                keep, extras = True, extras + 1
            if keep:
                unknown = [k for k in s["stats"] if k not in STAT_MAP][:2]
                stats.append(
                    {
                        **{k: s[k] for k in s if k != "stats"},
                        "stats": {
                            k: round(s["stats"][k], 4)
                            for k in [*STAT_MAP, *unknown]
                            if k in s["stats"]
                        },
                    }
                )
        slim_p = {k: p[k] for k in ("id", "fullName", "defaultPositionId")}
        players.append({"id": e["id"], "player": {**slim_p, "stats": stats}})
    return {"players": players}


def capture_espn() -> None:
    espn = EspnClient()
    for season, n in ((SEASON, 15), (2024, 10)):
        dump(OUT / "espn" / f"season_{season}.json", trim_espn(espn.season_raw(season), season, n))


def main() -> None:
    if sys.argv[1:] == ["espn"]:  # refresh just the ESPN fixtures (2 requests)
        return capture_espn()
    sl = SleeperClient()
    sdir = OUT / "sleeper"
    for name, rows in {
        "league": sl.league(LEAGUE),
        "rosters": sl.rosters(LEAGUE),
        "users": sl.users(LEAGUE),
        "state": sl.state(),
    }.items():
        dump(sdir / f"{name}.json", rows)
    for w in range(1, 18):
        dump(sdir / f"matchups_{w}.json", cache.get_json(f"{API}/league/{LEAGUE}/matchups/{w}"))
    for w in range(1, 8):
        dump(
            sdir / f"transactions_{w}.json",
            cache.get_json(f"{API}/league/{LEAGUE}/transactions/{w}"),
        )

    proj = {
        w: cache.get_json(f"{PROJ_API}/projections/nfl/{SEASON}/{w}", params=_pos_params())
        for w in range(7, 18)
    }
    rostered = {p for r in sl.rosters(LEAGUE) for p in (r["players"] or []) + (r["reserve"] or [])}
    keep = set(rostered)
    for pos in FANTASY_POSITIONS:
        rows = [r for r in proj[7] if r["player"]["position"] == pos]
        if pos == "DEF":
            keep |= {r["player_id"] for r in rows}
            continue
        rows.sort(key=lambda r: r["stats"].get("pts_ppr", 0), reverse=True)
        keep |= {r["player_id"] for r in rows[:TOP_N]}
    players = sl.players()
    keep |= {i for i, p in players.items() if p.get("position") == "DEF" and p.get("active")}

    dump(sdir / "players.json", {i: players[i] for i in sorted(keep) if i in players})
    for w, rows in proj.items():
        dump(sdir / f"projections_{SEASON}_{w}.json", slim(rows, keep))
    for w in range(1, 7):
        rows = cache.get_json(f"{PROJ_API}/stats/nfl/{SEASON}/{w}", params=_pos_params())
        dump(sdir / f"stats_{SEASON}_{w}.json", slim(rows, keep))

    fc = cache.get_json(
        "https://api.fantasycalc.com/values/current",
        params={"isDynasty": "false", "numQbs": 1, "numTeams": 14, "ppr": 1},
    )
    dump(OUT / "fantasycalc" / "values_14_1_1.json", fc)
    capture_espn()


if __name__ == "__main__":
    main()
