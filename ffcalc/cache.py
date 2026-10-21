"""Disk cache, weekly projection snapshots, and polite HTTP for every data source."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import polars as pl

log = logging.getLogger(__name__)

# Seconds between requests to the same host (spec §2 rate limits).
MIN_INTERVAL = {"lm-api-reads.fantasy.espn.com": 1.0}
DEFAULT_INTERVAL = 0.1

TTL = {
    "players": 24 * 3600,
    "projections": 3600,
    "league": 600,
    "fantasycalc": 6 * 3600,
    "nflverse": 24 * 3600,
    "history": 365 * 24 * 3600,  # finished weeks never change
}

_last_hit: dict[str, float] = {}
_lock = threading.Lock()


def data_dir() -> Path:
    return Path(os.environ.get("FFCALC_DATA", Path(__file__).resolve().parent.parent / "data"))


def _path(key: str, suffix: str) -> Path:
    p = data_dir() / "cache" / f"{key}{suffix}"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _tmp(p: Path) -> Path:
    # unique per write, so concurrent Streamlit threads never share a temp file
    return p.with_name(f".{p.name}.{uuid.uuid4().hex}.tmp")


def _fresh(p: Path, ttl: float) -> bool:
    return p.exists() and time.time() - p.stat().st_mtime < ttl


def cached_json(key: str, ttl: float, fetch: Callable[[], Any]) -> Any:
    """Return cached JSON for `key` if younger than `ttl`, else fetch and store.

    If the fetch fails and a stale copy exists, return the stale copy (offline fallback).
    """
    p = _path(key, ".json")
    if _fresh(p, ttl):
        return json.loads(p.read_text())
    try:
        value = fetch()
    except Exception:
        if p.exists():
            log.warning("fetch failed for %s; using stale cache", key)
            return json.loads(p.read_text())
        raise
    tmp = _tmp(p)
    tmp.write_text(json.dumps(value))
    tmp.replace(p)  # atomic: a crash never leaves a half-written cache file
    return value


def cached_frame(key: str, ttl: float, fetch: Callable[[], pl.DataFrame]) -> pl.DataFrame:
    """Same as cached_json for polars frames (stored as parquet)."""
    p = _path(key, ".parquet")
    if _fresh(p, ttl):
        return pl.read_parquet(p)
    try:
        df = fetch()
    except Exception:
        if p.exists():
            log.warning("fetch failed for %s; using stale cache", key)
            return pl.read_parquet(p)
        raise
    tmp = _tmp(p)
    df.write_parquet(tmp)
    tmp.replace(p)
    return df


def snapshot(source: str, season: int, week: int, df: pl.DataFrame) -> None:
    """Keep projections exactly as fetched, for future backtests (spec §2)."""
    p = data_dir() / "snapshots" / source / str(season) / f"{week}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(p)
    df.write_parquet(tmp)
    tmp.replace(p)  # latest pre-game fetch wins; see NOTES.md


def get_json(url: str, params: Any = None, headers: dict[str, str] | None = None) -> Any:
    """GET a JSON document, waiting so each host sees at most one request per interval."""
    host = urlparse(url).netloc
    interval = MIN_INTERVAL.get(host, DEFAULT_INTERVAL)
    with _lock:  # reserve a time slot for this host, then sleep outside the lock
        now = time.monotonic()
        slot = max(now, _last_hit.get(host, 0.0) + interval)
        _last_hit[host] = slot
    if slot > now:
        time.sleep(slot - now)
    r = httpx.get(
        url,
        params=params,
        headers={"User-Agent": "ffcalc/0.1 (personal fantasy tool)", **(headers or {})},
        timeout=30,
        follow_redirects=True,
    )
    r.raise_for_status()
    return r.json()
