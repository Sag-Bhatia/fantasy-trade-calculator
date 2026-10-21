"""Market values, the league's own trades and the acceptance heuristic (spec §8)."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

from .clients.fantasycalc import SCHEMA
from .models import League
if TYPE_CHECKING:  # type-only: trade.py imports this module lazily
    from .trade import Trade

log = logging.getLogger(__name__)

OVERPAY = 0.15  # A gives >15% more market value than it receives
K_TITLE = 0.5  # per percentage point of the partner's title-odds change
K_MARKET = 3.0  # per unit of the partner's market gap (0.10 = 10% of what it gives)
INTERCEPT = 0.0
P_MIN, P_MAX = 0.02, 0.98


@dataclass
class SideValue:
    players: list[str]  # sleeper ids
    values: dict[str, int]  # sleeper_id → FantasyCalc value (0 if unknown)
    total: int


@dataclass
class MarketCheck:
    a_gives: SideValue
    b_gives: SideValue
    gap: float  # (value A receives − value A gives) / max(value A gives, 1); negative = A overpays
    overpay: bool
    params: dict  # FantasyCalc params used: num_teams, num_qbs, ppr


@dataclass
class LeagueTrade:
    week: int
    created: int  # ms since epoch
    sides: list[tuple[int, list[str], list[str]]]  # (roster_id, received players, sent players)
    value_by_roster: dict[int, int]  # market value each roster received (today's values)


def fc_params(league: League) -> dict:
    superflex = "SUPER_FLEX" in league.slots or league.slots.count("QB") > 1
    rec = league.scoring.get("rec", 0)
    ppr = min((0, 0.5, 1), key=lambda p: abs(p - rec))
    return {"num_teams": league.num_teams, "num_qbs": 2 if superflex else 1, "ppr": ppr}


def values(src: Any, league: League) -> pl.DataFrame:
    """FantasyCalc values for this league's format; empty frame on any failure (optional data)."""
    try:
        return src.fantasycalc.values(**fc_params(league))
    except Exception as e:  # noqa: BLE001 — any client/network failure just disables the market view
        log.warning("FantasyCalc values unavailable: %s", e)
        return pl.DataFrame(schema=SCHEMA)


def _lookup(vals: pl.DataFrame) -> dict[str, int]:
    return dict(zip(vals["sleeper_id"].to_list(), vals["value"].to_list(), strict=True))


def _side(players: list[str], lookup: dict[str, int]) -> SideValue:
    v = {p: int(lookup.get(p) or 0) for p in players}
    return SideValue(list(players), v, sum(v.values()))


def market_check(trade: Trade, vals: pl.DataFrame, params: dict | None = None) -> MarketCheck:
    lk = _lookup(vals)
    a, b = _side(trade.a_gives, lk), _side(trade.b_gives, lk)
    # rounded so exactly 15% is not an overpay despite float error
    overpay = round(a.total - b.total * (1 + OVERPAY), 9) > 0
    return MarketCheck(a, b, (b.total - a.total) / max(a.total, 1), overpay, params or {})


def league_trades(src: Any, league: League, vals: pl.DataFrame) -> list[LeagueTrade]:
    """Completed trades so far this season, newest first.

    Draft picks are ignored (Sleeper lists them separately and FantasyCalc values players only),
    so a pick-heavy trade shows as lopsided."""
    lk = _lookup(vals)
    out = []
    for week in range(1, league.current_week + 1):
        for tx in src.sleeper.transactions(league.league_id, week) or []:
            if tx.get("type") != "trade" or tx.get("status") != "complete":
                continue
            adds, drops = tx.get("adds") or {}, tx.get("drops") or {}
            sides = [
                (
                    rid,
                    [p for p, r in adds.items() if r == rid],
                    [p for p, r in drops.items() if r == rid],
                )
                for rid in tx.get("roster_ids") or sorted(set(adds.values()) | set(drops.values()))
            ]
            worth = {rid: sum(int(lk.get(p) or 0) for p in got) for rid, got, _ in sides}
            out.append(LeagueTrade(week, int(tx.get("created") or 0), sides, worth))
    return sorted(out, key=lambda t: -t.created)


def acceptance(partner_title_delta: float, gap_for_partner: float) -> float:
    """HEURISTIC, not a measured probability (spec §8): how likely the partner is to say yes.

    logistic(K_TITLE * partner Δtitle odds (percentage points) + K_MARKET * partner's market gap),
    clipped to [0.02, 0.98]. K_TITLE = 0.5, K_MARKET = 3.0 and INTERCEPT = 0 are judgment calls,
    not fitted to any data: +2 points of title odds or a +33% market gap each move 0.5 to ~0.73.
    """
    z = INTERCEPT + K_TITLE * partner_title_delta + K_MARKET * gap_for_partner
    z = max(-50.0, min(50.0, z))  # keep exp() finite
    return min(P_MAX, max(P_MIN, 1 / (1 + math.exp(-z))))
