import os
import time

import polars as pl
import pytest

from ffcalc import cache


def test_cached_json_fetches_once_then_reuses():
    calls = []

    def fetch():
        calls.append(1)
        return {"a": 1}

    assert cache.cached_json("k", 60, fetch) == {"a": 1}
    assert cache.cached_json("k", 60, fetch) == {"a": 1}
    assert len(calls) == 1


def test_stale_cache_used_when_fetch_fails():
    cache.cached_json("k", 60, lambda: {"v": 1})
    p = cache.data_dir() / "cache" / "k.json"
    old = time.time() - 3600
    os.utime(p, (old, old))

    def boom():
        raise ConnectionError("down")

    assert cache.cached_json("k", 60, boom) == {"v": 1}


def test_fetch_error_without_cache_raises():
    def boom():
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        cache.cached_json("missing", 60, boom)


def test_cached_frame_roundtrip_and_snapshot():
    df = pl.DataFrame({"x": [1, 2]})
    assert cache.cached_frame("f", 60, lambda: df).equals(df)
    cache.snapshot("sleeper", 2026, 7, df)
    assert (cache.data_dir() / "snapshots" / "sleeper" / "2026" / "7.parquet").exists()


def test_network_is_blocked():
    with pytest.raises(RuntimeError):
        cache.get_json("https://api.sleeper.app/v1/state/nfl")


def test_expired_entry_is_refetched():
    cache.cached_json("k", 60, lambda: {"v": 1})
    p = cache.data_dir() / "cache" / "k.json"
    old = time.time() - 3600
    os.utime(p, (old, old))
    assert cache.cached_json("k", 60, lambda: {"v": 2}) == {"v": 2}


def test_rate_limit_spaces_requests(monkeypatch):
    slept = []
    monkeypatch.setattr(cache.time, "sleep", slept.append)

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {}

    monkeypatch.setattr(cache.httpx, "get", lambda *a, **k: R())
    url = "https://lm-api-reads.fantasy.espn.com/x"
    cache.get_json(url)
    cache.get_json(url)
    assert slept and slept[-1] > 0.5
