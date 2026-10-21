"""Load FantasyPros projection CSVs the user downloaded by hand (spec §2). Never scraped."""

from __future__ import annotations

import csv
import io
import logging
import re
from pathlib import Path
from typing import IO

import polars as pl

from ffcalc.clients.nflverse import current_season
from ffcalc.ids import normalize_name, normalize_team

log = logging.getLogger(__name__)

# (header, n-th occurrence of that header) → Sleeper stat key. FPTS is deliberately absent:
# value comes from our own scoring of stat lines, never from a source's points.
_MAPS: dict[str, dict[tuple[str, int], str]] = {
    "QB": {
        ("ATT", 0): "pass_att", ("CMP", 0): "pass_cmp", ("YDS", 0): "pass_yd",
        ("TDS", 0): "pass_td", ("INTS", 0): "pass_int", ("ATT", 1): "rush_att",
        ("YDS", 1): "rush_yd", ("TDS", 1): "rush_td", ("FL", 0): "fum_lost",
    },
    "RB": {
        ("ATT", 0): "rush_att", ("YDS", 0): "rush_yd", ("TDS", 0): "rush_td",
        ("REC", 0): "rec", ("YDS", 1): "rec_yd", ("TDS", 1): "rec_td", ("FL", 0): "fum_lost",
    },
    "WR": {
        ("REC", 0): "rec", ("YDS", 0): "rec_yd", ("TDS", 0): "rec_td",
        ("ATT", 0): "rush_att", ("YDS", 1): "rush_yd", ("TDS", 1): "rush_td",
        ("FL", 0): "fum_lost",
    },
    "TE": {("REC", 0): "rec", ("YDS", 0): "rec_yd", ("TDS", 0): "rec_td", ("FL", 0): "fum_lost"},
    "K": {("FG", 0): "fgm", ("FGA", 0): "fga", ("XPT", 0): "xpm"},
    "DEF": {
        ("SACK", 0): "sack", ("INT", 0): "int", ("FR", 0): "fum_rec", ("FF", 0): "ff",
        ("TD", 0): "def_td", ("SAFETY", 0): "safe", ("PA", 0): "pts_allow",
        ("YDS_AGN", 0): "yds_allow",
    },
}  # fmt: skip
_ALIASES = {"TD": "TDS", "INT": "INTS", "FGM": "FG", "XP": "XPT", "XPM": "XPT"}
_POS_HINT = {"DST": "DEF", "D/ST": "DEF"}

_STAT_COLS = {"stat": pl.String, "value": pl.Float64}
MATCHED_SCHEMA = {
    "source": pl.String, "season": pl.Int64, "week": pl.Int64, "sleeper_id": pl.String,
    **_STAT_COLS,
}  # fmt: skip
UNMATCHED_SCHEMA = {"name": pl.String, "team": pl.String, "position": pl.String}


# Sleeper's yds_allow_* bucket names, as seen in a real DEF projection (2026 wk 7). Other
# ranges exist in Sleeper but were not observed, so they are not emitted (no guessed names).
_YDS_BUCKETS = {(200, 299): "yds_allow_200_299", (300, 349): "yds_allow_300_349",
                (350, 399): "yds_allow_350_399", (400, 449): "yds_allow_400_449"}  # fmt: skip
_PTS_BUCKETS = [(0, "pts_allow_0"), (6, "pts_allow_1_6"), (13, "pts_allow_7_13"),
                (20, "pts_allow_14_20"), (27, "pts_allow_21_27"),
                (34, "pts_allow_28_34")]  # fmt: skip


def bucket_stats(pts_allow: float | None, yds_allow: float | None) -> dict[str, float]:
    """Sleeper bucket keys for a projected mean. shortcut: the bucket of the mean, not a
    distribution over buckets; upgrade when the simulator models points allowed directly."""
    out: dict[str, float] = {}
    if pts_allow is not None:
        p = round(pts_allow)
        out[next((k for hi, k in _PTS_BUCKETS if p <= hi), "pts_allow_35p")] = 1.0
    if yds_allow is not None:
        y = round(yds_allow)
        if k := next((k for (lo, hi), k in _YDS_BUCKETS.items() if lo <= y <= hi), None):
            out[k] = 1.0
    return out


def _dst_aliases(table: pl.DataFrame) -> dict[str, str]:
    """Normalized 'Denver Broncos' / 'Broncos' / 'Denver' / 'DEN' → team code; ambiguous dropped."""
    seen: dict[str, set[str]] = {}
    for sid, name in table.filter(pl.col("position") == "DEF").select("sleeper_id", "name").rows():
        words = name.split()
        for alias in (name, words[-1], " ".join(words[:-1]), sid):
            if alias:
                seen.setdefault(normalize_name(alias), set()).add(sid)
    return {a: next(iter(v)) for a, v in seen.items() if len(v) == 1}


def infer_position(headers: list[str]) -> str:
    """Guess the position from the stat columns. TE vs WR is ambiguous without rushing columns."""
    h = [x.upper() for x in headers]
    if "CMP" in h or "INTS" in h:
        return "QB"
    if "FGA" in h or "XPT" in h:
        return "K"
    if "SACK" in h or "SAFETY" in h:
        return "DEF"
    if "REC" in h:
        if "ATT" not in h:
            return "TE"  # shortcut: a WR export without rushing columns reads as TE, pass position
        return "RB" if h.index("ATT") < h.index("REC") else "WR"
    raise ValueError(f"cannot infer position from FantasyPros headers: {headers}")


def _read_rows(src: str | Path | IO[str] | IO[bytes]) -> list[list[str]]:
    if isinstance(src, str | Path):
        text = Path(src).read_text(encoding="utf-8-sig")
    else:
        raw = src.read()
        text = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw.lstrip("﻿")
    return [r for r in csv.reader(io.StringIO(text)) if any(c.strip() for c in r)]


def _num(x: str) -> float | None:
    try:
        return float(x.replace(",", "").strip())
    except ValueError:
        return None


def load(
    path_or_buffer: str | Path | IO[str] | IO[bytes],
    position: str | None,
    table: pl.DataFrame,
    *,
    season: int | None = None,
    week: int | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Parse one FantasyPros projection CSV → (matched stat lines, unmatched rows).

    Rows are matched to the ID table on normalized name + team + position and never guessed:
    ambiguous or unknown rows go to the second frame, as do repeats of an already-matched
    player (first row wins). DST rows need the name and the Team column to agree. A row with
    no team (e.g. a bare name) matches only if the name+position is unique in the table.
    DST rows also get Sleeper's pts_allow_*/yds_allow_* bucket keys, but only for weekly files
    (`week` given): a season total's bucket would be meaningless.
    """
    rows = _read_rows(path_or_buffer)
    if not rows:
        return pl.DataFrame(schema=MATCHED_SCHEMA), pl.DataFrame(schema=UNMATCHED_SCHEMA)
    headers = [h.strip().upper() for h in rows[0]]
    file_pos = _POS_HINT.get(position.upper(), position.upper()) if position else None
    file_pos = file_pos or infer_position(headers)
    if file_pos == "DEF":  # DST columns keep TD/INT; elsewhere they mean TDS/INTS
        norm = headers
    else:
        norm = [_ALIASES.get(h, h) for h in headers]
    seen: dict[str, int] = {}
    stat_cols: list[tuple[int, str]] = []
    col = {"PLAYER": -1, "TEAM": -1, "POS": -1}
    for i, h in enumerate(norm):
        n = seen.get(h, 0)
        seen[h] = n + 1
        if h in col and col[h] < 0:
            col[h] = i
        elif key := _MAPS[file_pos].get((h, n)):
            stat_cols.append((i, key))
    if col["PLAYER"] < 0:
        raise ValueError("FantasyPros CSV has no Player column")

    teams = {t for t in table["team"].to_list() if t}
    by_name: dict[tuple[str, str], list[tuple[str, str | None]]] = {}
    for tid, tpos, sname, tteam in table.select(
        "sleeper_id", "position", "search_name", "team"
    ).rows():
        by_name.setdefault((tpos, sname), []).append((tid, tteam))
    dst_alias = _dst_aliases(table)

    season = season or current_season()
    matched: list[dict] = []
    unmatched: list[dict] = []
    done: set[str] = set()
    for r in rows[1:]:
        r = r + [""] * (len(headers) - len(r))
        name = r[col["PLAYER"]].strip()
        team = normalize_team(r[col["TEAM"]]) if col["TEAM"] >= 0 else None
        if not name:
            continue
        if team is None:  # "Patrick Mahomes KC" style: peel a trailing known team code
            head, _, tail = name.rpartition(" ")
            if head and normalize_team(tail) in teams:
                name, team = head, normalize_team(tail)
        pos = file_pos
        if col["POS"] >= 0:  # "WR12"-style hint beats the file-level guess
            hint = r[col["POS"]].strip().upper().rstrip("0123456789")
            pos = _POS_HINT.get(hint, hint) or file_pos

        if pos == "DEF":  # the name must resolve to a franchise and agree with Team if given
            code = dst_alias.get(
                normalize_name(re.sub(r"\s+(D/?ST|defense)$", "", name, flags=re.I))
            )
            sid = code if code and team in (None, code) else None
        else:
            ids = [c[0] for c in by_name.get((pos, normalize_name(name)), [])
                   if team is None or c[1] == team]  # fmt: skip
            sid = ids[0] if len(ids) == 1 else None
        if sid is None:
            unmatched.append({"name": name, "team": team, "position": pos})
            continue
        if sid in done:  # a repeated player must not be counted twice
            log.warning("FantasyPros CSV repeats %s (%s); keeping the first row", name, sid)
            unmatched.append({"name": name, "team": team, "position": pos})
            continue
        done.add(sid)
        vals = {key: v for i, key in stat_cols if (v := _num(r[i])) is not None}
        if pos == "DEF" and week is not None:
            vals |= bucket_stats(vals.get("pts_allow"), vals.get("yds_allow"))
        for key, v in vals.items():
            matched.append(
                {"source": "fantasypros", "season": season, "week": week,
                 "sleeper_id": sid, "stat": key, "value": v}
            )  # fmt: skip
    return (
        pl.DataFrame(matched, schema=MATCHED_SCHEMA),
        pl.DataFrame(unmatched, schema=UNMATCHED_SCHEMA),
    )
