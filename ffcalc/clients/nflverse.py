"""nflverse data via nflreadpy, cached per season (spec §2)."""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Callable
from functools import partial

import nflreadpy
import polars as pl

from ffcalc import cache

log = logging.getLogger(__name__)


def current_season(today: dt.date | None = None) -> int:
    today = today or dt.date.today()
    return today.year if today.month >= 3 else today.year - 1


class NflverseClient:
    def _seasons(
        self, name: str, seasons: list[int], load: Callable[[int], pl.DataFrame]
    ) -> pl.DataFrame:
        """One cache entry per season so a season published later is picked up on its own."""
        frames = []
        for s in seasons:
            ttl = cache.TTL["history" if s < current_season() else "nflverse"]
            try:
                frames.append(cache.cached_frame(f"nflverse/{name}/{s}", ttl, partial(load, s)))
            except ValueError as e:  # nflreadpy: season not published yet; anything else raises
                log.warning("nflverse %s %s unavailable: %s", name, s, e)
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="diagonal_relaxed")

    def player_stats(self, seasons: list[int]) -> pl.DataFrame:
        """Weekly regular-season player stats."""
        return self._seasons(
            "player_stats",
            seasons,
            lambda s: nflreadpy.load_player_stats([s]).filter(pl.col("season_type") == "REG"),
        )

    def schedules(self, seasons: list[int]) -> pl.DataFrame:
        return self._seasons("schedules", seasons, lambda s: nflreadpy.load_schedules([s]))

    def injuries(self, seasons: list[int]) -> pl.DataFrame:
        return self._seasons("injuries", seasons, lambda s: nflreadpy.load_injuries([s]))

    def snap_counts(self, seasons: list[int]) -> pl.DataFrame:
        return self._seasons("snap_counts", seasons, lambda s: nflreadpy.load_snap_counts([s]))

    def ff_playerids(self) -> pl.DataFrame:
        return cache.cached_frame(
            "nflverse/ff_playerids", cache.TTL["nflverse"], nflreadpy.load_ff_playerids
        )
