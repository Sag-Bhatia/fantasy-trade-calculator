"""Optimal legal lineups (spec §5).

`optimal_lineup` is exact for any slot list (DP over remaining slot counts).
`lineup_points` is the vectorized version used inside the simulator: dedicated slots
first, then flex slots from the narrowest eligibility to the widest. That greedy order
is exact when flex eligibility sets are nested or disjoint (standard, superflex). When
two flex types cross (WRRB_FLEX with REC_FLEX) every order of the flex groups is tried
and the best kept per sim; tests check all of it against `optimal_lineup`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from functools import cache
from itertools import permutations

import numpy as np

from .models import POSITIONS, SLOT_ELIGIBILITY

POS_INDEX = {p: i for i, p in enumerate(POSITIONS)}


def slot_groups(slots: Sequence[str]) -> list[tuple[str, int, frozenset[int]]]:
    """[(slot, count, eligible position indexes)] in greedy fill order."""
    counts = Counter(s for s in slots if s in SLOT_ELIGIBILITY)
    groups = [
        (s, n, frozenset(POS_INDEX[p] for p in SLOT_ELIGIBILITY[s])) for s, n in counts.items()
    ]
    return sorted(groups, key=lambda g: (len(g[2]), g[0]))


def optimal_lineup(
    expected: Sequence[float], pos: Sequence[int], slots: Sequence[str]
) -> tuple[float, list[tuple[int, str]]]:
    """Exact best lineup. Returns (total, [(player index, slot)]). Empty slots score 0."""
    groups = slot_groups(slots)
    order = sorted(range(len(expected)), key=lambda i: -expected[i])
    names = [g[0] for g in groups]
    elig = [g[2] for g in groups]

    @cache
    def best(k: int, remaining: tuple[int, ...]) -> tuple[float, tuple[tuple[int, int], ...]]:
        if k == len(order) or not any(remaining):
            return 0.0, ()
        i = order[k]
        result = best(k + 1, remaining)  # bench player i
        if expected[i] > 0:
            for g, left in enumerate(remaining):
                if left and pos[i] in elig[g]:
                    rest = remaining[:g] + (left - 1,) + remaining[g + 1 :]
                    v, picks = best(k + 1, rest)
                    if v + expected[i] > result[0]:
                        result = (v + expected[i], ((i, g), *picks))
        return result

    total, picks = best(0, tuple(g[1] for g in groups))
    return total, [(i, names[g]) for i, g in picks]


def _crossing(groups: list[tuple[str, int, frozenset[int]]]) -> bool:
    sets = [g[2] for g in groups if len(g[2]) > 1]
    return any(a & b and not (a <= b or b <= a) for i, a in enumerate(sets) for b in sets[i + 1 :])


def lineup_points(
    scores: np.ndarray,
    available: np.ndarray,
    expected: np.ndarray,
    pos: np.ndarray,
    slots: Sequence[str],
    priority: np.ndarray | None = None,
    free_agent: np.ndarray | None = None,
    max_free_agents: int = 1,
) -> np.ndarray:
    """Vectorized lineup scoring over sims for one team-week.

    scores, available: (sims, n) realized points and "manager knows he can start him";
    expected: (n,) points used to choose the lineup (never realized points); pos: (n,).
    priority: optional (n,) sort key overriding `expected` (e.g. rostered players before
    free agents, so a free agent only fills a slot nobody on the roster can).
    free_agent: optional (n,) bool; at most `max_free_agents` of them start per sim.
    Returns (sims,) starting-lineup points.
    """
    order = np.argsort(-(expected if priority is None else priority), kind="stable")
    sc = scores[:, order]
    av = available[:, order] & (expected[order] > 0)
    ps = pos[order]
    fa = np.zeros(len(order), bool) if free_agent is None else free_agent[order]
    groups = slot_groups(slots)
    dedicated = [g for g in groups if len(g[2]) == 1]
    flex = [g for g in groups if len(g[2]) > 1]
    orders = list(permutations(flex)) if _crossing(groups) else [tuple(flex)]
    best = None
    for flex_order in orders:
        chosen = np.zeros_like(av)
        fa_used = np.zeros(av.shape[0], np.int64)
        for _, count, elig in [*dedicated, *flex_order]:
            cand = av & ~chosen & np.isin(ps, list(elig))[None, :]
            room = (max_free_agents - fa_used)[:, None]
            cand &= ~fa[None, :] | (np.cumsum(cand & fa[None, :], axis=1) <= room)
            take = cand & (np.cumsum(cand, axis=1) <= count)
            chosen |= take
            fa_used += (take & fa[None, :]).sum(axis=1)
        pts = (sc * chosen).sum(axis=1)
        best = pts if best is None else np.maximum(best, pts)
    assert best is not None
    return best
