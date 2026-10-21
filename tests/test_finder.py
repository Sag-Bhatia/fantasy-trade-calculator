import os
import tempfile
from collections import Counter

import pytest

from ffcalc import engine, finder
from tests.fakes import fake_sources


@pytest.fixture(scope="module")
def ctx():
    os.environ["FFCALC_DATA"] = tempfile.mkdtemp()
    return engine.prepare("1400221038771437568", src=fake_sources(), build_history=False)


def test_candidates_stay_in_value_band_and_skip_k_def(ctx):
    ros = finder._ros(ctx)
    val = finder._value(ctx)
    cands = finder.candidates(ctx, 1)
    assert cands
    for t in cands[:400]:
        assert t.team_a == 1 and t.team_b != 1
        assert 1 <= len(t.a_gives) <= 2 and 1 <= len(t.b_gives) <= 2
        for p in t.a_gives + t.b_gives:
            assert ctx.inputs.pos[ctx.index[p]] not in (4, 5)
        if val and all(val.get(p, 0) > 0 for p in t.a_gives + t.b_gives):
            g = sum(val[p] for p in t.a_gives)
            r = sum(val[p] for p in t.b_gives)
        else:
            g = sum(ros[ctx.index[p]] for p in t.a_gives)
            r = sum(ros[ctx.index[p]] for p in t.b_gives)
        assert abs(r - g) / max(g, r) <= finder.BAND + 1e-9


def test_need_filter_keeps_only_mutual_lineup_gains(ctx):
    kept = finder._need_filter(ctx, finder.candidates(ctx, 1)[:300])
    assert all(score > 0 for score, _ in kept)
    assert [s for s, _ in kept] == sorted((s for s, _ in kept), reverse=True)


@pytest.mark.slow
def test_find_trades_returns_only_mutually_beneficial(ctx, monkeypatch):
    monkeypatch.setattr(finder, "PER_TEAM_SCREEN", 2)
    monkeypatch.setattr(finder, "CONFIRM_KEEP", 6)
    found = finder.find_trades(ctx, 1, screen_sims=300, confirm_sims=1000)
    for s in found:
        assert s.a_title.value > s.a_title.se and s.b_title.value > s.b_title.se
    per = Counter(s.trade.team_b for s in found)
    assert all(n <= finder.MAX_PER_TEAM for n in per.values())
    trades = [s.trade for s in found]
    assert not any(finder._variant(t, trades[:i]) for i, t in enumerate(trades))
    mins = [min(s.a_title.value, s.b_title.value) for s in found]
    assert mins == sorted(mins, reverse=True)


def test_diverse_drops_throw_in_variants_and_caps_each_partner():
    T = finder.Trade
    ranked = [
        T(1, 2, ["a", "b"], ["x"]),
        T(1, 2, ["a", "b"], ["x", "y"]),  # same core deal plus a throw-in
        T(1, 2, ["c"], ["z"]),  # a different deal with the same team
        T(1, 2, ["d"], ["w"]),  # third deal with team 2: over the cap
        T(1, 3, ["a"], ["q"]),  # same player of mine, different partner: fine
    ]
    picked = finder.diverse(ranked, n=10, per_team=2)
    assert picked == [ranked[0], ranked[2], ranked[4]]
