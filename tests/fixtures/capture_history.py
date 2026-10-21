"""Manual: `uv run python tests/fixtures/capture_history.py` — real 2025 weeks 1-4 Sleeper
projections + actuals, trimmed to fantasy-relevant players, for offline history tests."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ffcalc  # noqa: F401,E402  (TLS setup)
from ffcalc import cache  # noqa: E402
from ffcalc.clients.sleeper import PROJ_API, _pos_params  # noqa: E402

OUT = Path(__file__).parent / "sleeper"
KEEP_ROW = ("player_id", "season", "week", "team", "opponent", "game_id", "stats")


def trim(r: dict) -> dict:
    out = {k: r.get(k) for k in KEEP_ROW}
    out["player"] = {"position": (r.get("player") or {}).get("position")}
    out["stats"] = {
        k: round(v, 3)
        for k, v in (r.get("stats") or {}).items()
        if isinstance(v, int | float) and v and not k.startswith(("adp", "pos_", "rank"))
    }
    return out


for week in range(1, 5):
    proj = cache.get_json(f"{PROJ_API}/projections/nfl/2025/{week}", params=_pos_params())
    stats = cache.get_json(f"{PROJ_API}/stats/nfl/2025/{week}", params=_pos_params())
    keep = {
        r["player_id"]
        for r in proj
        if (r["stats"].get("pts_ppr") or 0) >= 3
        or (r.get("player") or {}).get("position") in ("K", "DEF")
    }
    for name, rows in (("projections", proj), ("stats", stats)):
        rows = [trim(r) for r in rows if r["player_id"] in keep]
        (OUT / f"{name}_2025_{week}.json").write_text(json.dumps(rows, separators=(",", ":")))
        print(name, week, len(rows))
