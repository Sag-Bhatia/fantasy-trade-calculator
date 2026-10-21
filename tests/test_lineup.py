import itertools

import numpy as np
import pytest

from ffcalc.lineup import POS_INDEX, lineup_points, optimal_lineup, slot_groups
from ffcalc.models import SLOT_ELIGIBILITY

STANDARD = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF"]
SUPERFLEX = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "FLEX", "SUPER_FLEX"]
ODD = ["QB", "RB", "WR", "TE", "WRRB_FLEX", "REC_FLEX", "SUPER_FLEX"]


def brute_force(expected, pos, slots):
    """Try every assignment of players to slots."""
    slots = [s for s in slots if s in SLOT_ELIGIBILITY]
    best = 0.0
    n = len(expected)
    for perm in itertools.product(range(-1, n), repeat=len(slots)):
        used = [i for i in perm if i >= 0]
        if len(used) != len(set(used)):
            continue
        total = 0.0
        ok = True
        for i, s in zip(perm, slots, strict=True):
            if i < 0:
                continue
            if pos[i] not in {POS_INDEX[p] for p in SLOT_ELIGIBILITY[s]}:
                ok = False
                break
            total += max(expected[i], 0)
        if ok:
            best = max(best, total)
    return best


def random_roster(rng, n):
    pos = rng.choice(list(POS_INDEX.values()), size=n, p=[0.15, 0.25, 0.3, 0.15, 0.075, 0.075])
    expected = np.round(rng.gamma(2.0, 5.0, size=n), 2)
    expected[rng.random(n) < 0.1] = 0.0  # byes / out
    return expected, pos


@pytest.mark.parametrize(
    "slots",
    [["QB", "RB", "WR", "FLEX"], ["RB", "WR", "TE", "SUPER_FLEX"],
     ["WR", "WRRB_FLEX", "REC_FLEX"], ["WRRB_FLEX", "REC_FLEX", "FLEX"]],
)  # fmt: skip
def test_dp_matches_brute_force(slots):
    rng = np.random.default_rng(0)
    for _ in range(150):
        expected, pos = random_roster(rng, int(rng.integers(3, 7)))
        total, picks = optimal_lineup(list(expected), list(pos), slots)
        assert total == pytest.approx(brute_force(expected, pos, slots))
        assert len({i for i, _ in picks}) == len(picks)


@pytest.mark.parametrize("slots", [STANDARD, SUPERFLEX, ODD])
def test_vectorized_matches_dp_on_500_rosters(slots):
    rng = np.random.default_rng(1)
    for _ in range(500):
        n = int(rng.integers(8, 20))
        expected, pos = random_roster(rng, n)
        available = rng.random((1, n)) > 0.15
        e = np.where(available[0], expected, 0.0)
        want, _ = optimal_lineup(list(e), list(pos), slots)
        got = lineup_points(expected[None, :], available, expected, pos, slots)[0]
        assert got == pytest.approx(want)


def test_lineup_uses_expected_not_realized_points():
    pos = np.array([POS_INDEX["QB"], POS_INDEX["QB"]])
    expected = np.array([20.0, 10.0])
    realized = np.array([[5.0, 30.0]])  # the backup went off, but you started the starter
    out = lineup_points(realized, np.ones((1, 2), bool), expected, pos, ["QB"])
    assert out[0] == 5.0


def test_unavailable_starter_is_replaced_and_empty_slot_scores_zero():
    pos = np.array([POS_INDEX["RB"], POS_INDEX["RB"], POS_INDEX["WR"]])
    expected = np.array([15.0, 8.0, 12.0])
    scores = np.tile([15.0, 8.0, 12.0], (2, 1))
    avail = np.array([[True, True, True], [False, True, False]])
    out = lineup_points(scores, avail, expected, pos, ["RB", "WR", "TE"])
    assert out.tolist() == [27.0, 8.0]


def test_slot_groups_orders_narrow_to_wide():
    names = [g[0] for g in slot_groups(SUPERFLEX)]
    assert names.index("FLEX") < names.index("SUPER_FLEX")
    assert names.index("QB") < names.index("FLEX")


def test_crossing_flex_counterexample():
    pos = np.array([POS_INDEX["WR"], POS_INDEX["TE"], POS_INDEX["RB"]])
    e = np.array([10.0, 9.0, 1.0])
    got = lineup_points(e[None, :], np.ones((1, 3), bool), e, pos, ["WRRB_FLEX", "REC_FLEX"])
    assert got[0] == 19.0


def test_free_agents_fill_only_empty_slots_one_per_week():
    rb, wr = POS_INDEX["RB"], POS_INDEX["WR"]
    pos = np.array([rb, rb, rb, wr])  # rostered RB (on bye), FA RB 10, FA RB 9, rostered WR
    e = np.array([0.0, 10.0, 9.0, 8.0])
    fa = np.array([False, True, True, False])
    out = lineup_points(e[None, :], np.ones((1, 4), bool), e, pos, ["RB", "RB", "WR"],
                        priority=e + (~fa) * 1e4, free_agent=fa)  # fmt: skip
    assert out[0] == 18.0  # one pickup (the 10-pt RB) + the rostered WR; second RB slot empty
