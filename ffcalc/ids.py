"""The one player-identity table: Sleeper players joined to the nflverse ID map (spec §2)."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import polars as pl

from ffcalc.models import POSITIONS

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

# Team codes other sources use → Sleeper's code.
TEAM_ALIASES = {
    "JAC": "JAX", "WSH": "WAS", "LA": "LAR", "ARZ": "ARI", "LVR": "LV", "OAK": "LV",
    "SD": "LAC", "STL": "LAR", "KCC": "KC", "NEP": "NE", "GBP": "GB", "NOS": "NO",
    "SFO": "SF", "TBB": "TB", "BLT": "BAL", "CLV": "CLE", "HST": "HOU",
}  # fmt: skip

_ID_COLS = ["espn_id", "gsis_id", "fantasypros_id"]
SCHEMA: dict[str, Any] = {
    "sleeper_id": pl.String,
    "name": pl.String,
    "position": pl.String,
    "team": pl.String,
    "injury_status": pl.String,
    "status": pl.String,
    "age": pl.Int64,
    "years_exp": pl.Int64,
    "espn_id": pl.String,
    "gsis_id": pl.String,
    "fantasypros_id": pl.String,
    "search_name": pl.String,
}


def normalize_name(name: str) -> str:
    """Lowercase, no accents/punctuation/spaces, trailing jr/sr/ii/iii/iv/v dropped."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = s.replace(".", "").replace("'", "")
    tokens = re.sub(r"[^a-z0-9]+", " ", s).split()
    while len(tokens) > 1 and tokens[-1] in _SUFFIXES:
        tokens.pop()
    return "".join(tokens)


def normalize_team(team: str | None) -> str | None:
    if not team:
        return None
    t = team.strip().upper()
    return TEAM_ALIASES.get(t, t)


def _s(x: Any) -> str | None:
    return None if x is None or x == "" else str(x)


def player_table(sleeper_players: dict[str, dict], idmap: pl.DataFrame) -> pl.DataFrame:
    """One row per Sleeper player at a fantasy position (DEF rows keyed by team code)."""
    rows = []
    for pid, p in sleeper_players.items():
        if p.get("position") not in POSITIONS:
            continue
        name = p.get("full_name") or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
        rows.append(
            {
                "sleeper_id": str(pid),
                "name": name,
                "position": p["position"],
                "team": normalize_team(p.get("team")),
                "injury_status": p.get("injury_status"),
                "status": p.get("status"),
                "age": p.get("age"),
                "years_exp": p.get("years_exp"),
                # Sleeper's own ids are the fallback when the idmap lacks them
                "s_espn_id": _s(p.get("espn_id")),
                "s_gsis_id": _s(p.get("gsis_id")),
                "search_name": normalize_name(name),
            }
        )
    base = pl.DataFrame(
        rows,
        schema={**SCHEMA, "s_espn_id": pl.String, "s_gsis_id": pl.String},
    )
    ids = (
        idmap.filter(pl.col("sleeper_id").is_not_null())
        .select(
            pl.col("sleeper_id").cast(pl.Int64).cast(pl.String),
            pl.col("espn_id").cast(pl.Int64, strict=False).cast(pl.String),
            pl.col("gsis_id").cast(pl.String),
            pl.col("fantasypros_id").cast(pl.Int64, strict=False).cast(pl.String),
        )
        .unique("sleeper_id", keep="first")
        .rename({c: f"m_{c}" for c in _ID_COLS})
    )
    out = base.join(ids, on="sleeper_id", how="left").with_columns(
        pl.coalesce("m_espn_id", "s_espn_id").alias("espn_id"),
        pl.coalesce("m_gsis_id", "s_gsis_id").alias("gsis_id"),
        pl.col("m_fantasypros_id").alias("fantasypros_id"),
    )
    return out.select(list(SCHEMA))


def lookup(table: pl.DataFrame, query: str) -> str | None:
    """Sleeper id, or a name (normalized: exact, then unique prefix) → sleeper_id; else None."""
    q = query.strip()
    if not q:
        return None
    if table.filter(pl.col("sleeper_id") == q).height:
        return q
    n = normalize_name(q)
    if not n:
        return None
    for hits in (
        table.filter(pl.col("search_name") == n),
        table.filter(pl.col("search_name").str.starts_with(n)),
    ):
        for narrow in (pl.col("status") == "Active", pl.col("team").is_not_null()):
            if hits.height > 1 and hits.filter(narrow).height:  # namesakes: prefer active, rostered
                hits = hits.filter(narrow)
        if hits.height == 1:
            return str(hits["sleeper_id"][0])
        if hits.height > 1:
            return None
    return None
