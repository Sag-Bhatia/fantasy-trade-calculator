"""Monte Carlo rest-of-season simulator (spec §6).

Two stages:
1. `simulate_players` draws every player's weekly outcome for a fixed player universe:
   correlated within NFL games (Gaussian copula by role), skewed (empirical residual
   quantiles by position × projection tier), game-time inactives (`p_play`), and
   in-season injuries that keep a player out for several weeks.
2. `simulate_league` scores any roster configuration against those outcomes, plays the
   remaining regular season (wins, ties, median wins, points-for tiebreaker) and the
   playoff bracket (byes, one- or two-week rounds).

Player outcomes don't depend on who rosters a player, so the with-trade and
without-trade runs share every random draw: common random numbers by construction.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.special import ndtr

from .lineup import lineup_points
from .models import POSITIONS, ROLES, League

NPOS = len(POSITIONS)
NROLES = len(ROLES)
QGRID = 101  # residual quantiles stored at u = 0, 0.01, ..., 1
FA_PER_SLOT = 3  # free agents per position considered for an empty slot


@dataclass
class Tables:
    """History-derived distribution tables (built by ffcalc.history)."""

    tier_edges: np.ndarray  # (NPOS, T-1) projection cut points per position
    resid_q: np.ndarray  # (NPOS, T, QGRID) residual (actual - projection) quantiles
    floor: np.ndarray  # (NPOS,) lowest realistic weekly score
    corr_same: np.ndarray  # (NROLES, NROLES) residual correlation, same NFL team
    corr_opp: np.ndarray  # (NROLES, NROLES) residual correlation, opponents
    hazard: np.ndarray  # (NPOS,) chance per game a player gets hurt and misses later weeks
    duration_pmf: np.ndarray  # (NPOS, D) P(misses exactly d+1 more weeks)
    source: str = "placeholder"
    # game status → per-position P(plays) and mean multiplier if he plays
    # placeholder defaults until history.status_tables runs on nflverse injury reports
    status_play: dict[str, list[float]] = field(
        default_factory=lambda: {
            "Questionable": [0.85] * NPOS,
            "Doubtful": [0.25] * NPOS,
            "Out": [0.0] * NPOS,
        }  # fmt: skip
    )
    status_mult: dict[str, list[float]] = field(
        default_factory=lambda: {
            "Questionable": [0.95] * NPOS,
            "Doubtful": [0.85] * NPOS,
            "Out": [1.0] * NPOS,
        }  # fmt: skip
    )


@dataclass
class SimInputs:
    """Per-player, per-week inputs for one league's universe of players."""

    ids: list[str]  # universe: every rostered player + the free-agent pool
    pos: np.ndarray  # (P,) position index
    weeks: list[int]  # simulated NFL weeks
    mean: np.ndarray  # (P, W) expected points if he plays (league scoring)
    p_play: np.ndarray  # (P, W) chance he suits up (0 = bye / ruled out)
    game: np.ndarray  # (P, W) game index within the week, -1 = no game
    team: np.ndarray  # (P, W) NFL team index, -1 = none
    role: np.ndarray  # (P, W) correlation role index, -1 = uncorrelated
    out_weeks: np.ndarray = field(default_factory=lambda: np.zeros(0, int))  # (P,) known out
    healthy_mean: np.ndarray | None = None  # (P, W) mean before injury-status adjustments

    @property
    def expected(self) -> np.ndarray:
        return self.mean * self.p_play


@dataclass
class Outcomes:
    points: np.ndarray  # (S, W, P) float32, 0 when he didn't play
    known_out: np.ndarray  # (S, W, P) bool, the manager knows before lineup lock
    expected: np.ndarray  # (P, W) points managers use to set lineups


def _tiers(tables: Tables, pos: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """(P, W) flat bucket index pos * T + tier."""
    t = tables.resid_q.shape[1]
    tier = np.zeros(mean.shape, dtype=np.intp)
    for p in range(NPOS):
        rows = pos == p
        tier[rows] = np.searchsorted(tables.tier_edges[p], mean[rows], side="right")
    return pos[:, None] * t + tier


def _game_factors(inp: SimInputs, tables: Tables, w: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """[(player indexes, Cholesky factor)] for each NFL game in week column w."""
    out = []
    for g in np.unique(inp.game[:, w]):
        if g < 0:
            continue
        idx = np.flatnonzero((inp.game[:, w] == g) & (inp.role[:, w] >= 0))
        if len(idx) < 2:
            continue
        r, tm = inp.role[idx, w], inp.team[idx, w]
        same = tm[:, None] == tm[None, :]
        c = np.where(same, tables.corr_same[r][:, r], tables.corr_opp[r][:, r])
        np.fill_diagonal(c, 1.0)
        vals, vecs = np.linalg.eigh(c)  # repair to the nearest valid correlation matrix
        c = (vecs * np.clip(vals, 1e-6, None)) @ vecs.T
        d = np.sqrt(np.diag(c))
        out.append((idx, np.linalg.cholesky(c / d[:, None] / d[None, :]).astype(np.float32)))
    return out


def simulate_players(inp: SimInputs, tables: Tables, sims: int, seed: int) -> Outcomes:
    n_players, n_weeks = inp.mean.shape
    # one stream per player keyed by his id: his draws don't depend on who else is simulated
    gens = [np.random.default_rng([seed, zlib.crc32(pid.encode())]) for pid in inp.ids]
    bucket = _tiers(tables, inp.pos, inp.mean)
    q = tables.resid_q.reshape(-1, QGRID).astype(np.float32)
    floor = tables.floor[inp.pos].astype(np.float32)
    hazard = tables.hazard[inp.pos].astype(np.float32)
    dur_cdf = np.cumsum(tables.duration_pmf, axis=1)[inp.pos]  # (P, D)
    out_weeks = inp.out_weeks if len(inp.out_weeks) else np.zeros(n_players, int)

    points = np.zeros((sims, n_weeks, n_players), np.float32)
    known_out = np.zeros((sims, n_weeks, n_players), bool)
    out_left = np.broadcast_to(out_weeks, (sims, n_players)).copy()

    for w in range(n_weeks):
        # results depend only on (seed, player id, sim, week)
        z = np.empty((sims, n_players), np.float32)
        u3 = np.empty((3, sims, n_players), np.float32)
        for i, g in enumerate(gens):
            z[:, i] = g.standard_normal(sims, dtype=np.float32)
            u3[:, :, i] = g.random((3, sims), dtype=np.float32)
        u_play, u_hurt, u_dur = u3

        for idx, chol in _game_factors(inp, tables, w):
            z[:, idx] = z[:, idx] @ chol.T
        u = ndtr(z) * (QGRID - 1)
        lo = np.minimum(u.astype(np.intp), QGRID - 2)
        frac = (u - lo).astype(np.float32)
        b = bucket[:, w][None, :]
        resid = q[b, lo] * (1 - frac) + q[b, lo + 1] * frac

        ruled_out = (out_left > 0) | (inp.p_play[:, w] <= 0)[None, :]
        plays = ~ruled_out & (u_play < inp.p_play[:, w][None, :])
        score = np.maximum(inp.mean[:, w][None, :] + resid, floor[None, :])
        points[:, w] = np.where(plays, score, 0.0)
        known_out[:, w] = ruled_out

        hurt = plays & (u_hurt < hazard[None, :])
        dur = 1 + (u_dur[:, :, None] > dur_cdf[None, :, :]).sum(axis=2)
        # absences count his team's games, so the clock stops on a bye
        ticks = ((inp.game[:, w] >= 0) | (inp.mean[:, w] > 0))[None, :]
        out_left = np.where(hurt, dur, np.where(ticks, np.maximum(out_left - 1, 0), out_left))

    return Outcomes(points, known_out, inp.expected.astype(np.float32))


# --------------------------------------------------------------------------- league


@dataclass
class Config:
    """Who rosters whom: team index → universe indexes. FA pool = candidates for empty slots."""

    rosters: list[np.ndarray]
    fa_pool: np.ndarray


@dataclass
class SeasonResult:
    team_points: np.ndarray  # (S, N, W) starting-lineup points
    made_playoffs: np.ndarray  # (S, N) bool
    champion: np.ndarray  # (S, N) bool
    seed: np.ndarray  # (S, N) 1-based final seed
    wins: np.ndarray  # (S, N) regular-season wins incl. played weeks

    @property
    def playoff_prob(self) -> np.ndarray:
        return self.made_playoffs.mean(axis=0)

    @property
    def title_prob(self) -> np.ndarray:
        return self.champion.mean(axis=0)


def team_points(out: Outcomes, inp: SimInputs, cfg: Config, slots: Sequence[str]) -> np.ndarray:
    """(S, N, W) starting-lineup points for every team."""
    sims, n_weeks, _ = out.points.shape
    res = np.zeros((sims, len(cfg.rosters), n_weeks), np.float32)
    for t, roster in enumerate(cfg.rosters):
        res[:, t] = team_points_one(out, inp, roster, cfg.fa_pool, slots)
    return res


def team_points_one(
    out: Outcomes, inp: SimInputs, roster: np.ndarray, fa_pool: np.ndarray, slots: Sequence[str]
) -> np.ndarray:
    """(S, W) for one roster. Free agents only fill slots the roster can't, at most one
    pickup per team-week (spec §5)."""
    fa = np.setdiff1d(fa_pool, roster)
    # only the best few free agents per position can ever be the one pickup
    best = []
    for p in range(NPOS):
        at_pos = fa[inp.pos[fa] == p]
        best.append(at_pos[np.argsort(-out.expected[at_pos].sum(1))[:FA_PER_SLOT]])
    fa = np.concatenate(best) if best else fa
    cand = np.concatenate([roster, fa]).astype(np.intp)
    rostered = np.r_[np.ones(len(roster)), np.zeros(len(fa))]
    res = np.zeros((out.points.shape[0], out.points.shape[1]), np.float32)
    for w in range(out.points.shape[1]):
        e = out.expected[cand, w]
        res[:, w] = lineup_points(
            out.points[:, w, cand],
            ~out.known_out[:, w, cand],
            e,
            inp.pos[cand],
            slots,
            priority=e + rostered * 1e4,
            free_agent=rostered == 0,
        )
    return res


def playoff_rounds(league: League) -> list[list[int]]:
    """NFL weeks belonging to each playoff round."""
    n_rounds = max(1, math.ceil(math.log2(max(league.playoff_teams, 2))))
    lengths = [1] * n_rounds
    if league.playoff_round_type == 1:
        lengths[-1] = 2
    elif league.playoff_round_type == 2:
        lengths = [2] * n_rounds
    rounds, week = [], league.playoff_week_start
    for n in lengths:
        rounds.append(list(range(week, week + n)))
        week += n
    return rounds


def bracket_order(size: int) -> list[int]:
    """Seeds (1-based) in bracket slot order, e.g. 8 → [1, 8, 4, 5, 2, 7, 3, 6]."""
    order = [1]
    while len(order) < size:
        m = 2 * len(order) + 1
        order = [s for x in order for s in (x, m - x)]
    return order


def simulate_league(
    out: Outcomes, inp: SimInputs, league: League, cfg: Config, tp: np.ndarray | None = None
) -> SeasonResult:
    """Play the rest of the season. `tp` lets callers reuse precomputed team points."""
    if tp is None:
        tp = team_points(out, inp, cfg, league.slots)
    sims, n, _ = tp.shape
    col = {wk: i for i, wk in enumerate(inp.weeks)}
    rid = [t.roster_id for t in league.teams]
    index = {r: i for i, r in enumerate(rid)}

    wins = np.tile(np.array([t.wins + 0.5 * t.ties for t in league.teams], np.float32), (sims, 1))
    pf = np.tile(np.array([t.points_for for t in league.teams], np.float32), (sims, 1))
    for wk, pairs in sorted(league.schedule.items()):
        if wk < league.current_week or wk not in col:
            continue
        pts = tp[:, :, col[wk]]
        a = np.array([index[x] for x, _ in pairs])
        b = np.array([index[y] for _, y in pairs])
        wins[:, a] += (pts[:, a] > pts[:, b]) + 0.5 * (pts[:, a] == pts[:, b])
        wins[:, b] += (pts[:, b] > pts[:, a]) + 0.5 * (pts[:, a] == pts[:, b])
        if league.median_win:
            med = np.median(pts, axis=1, keepdims=True)
            wins += (pts > med) + 0.5 * (pts == med)
        pf += pts

    # seeding: wins, then points for (Sleeper default tiebreaker)
    key = wins.astype(np.float64) + pf.astype(np.float64) * 1e-6
    by_seed = np.argsort(-key, axis=1, kind="stable")  # (S, N) team index per seed
    seed = np.empty_like(by_seed)
    np.put_along_axis(seed, by_seed, np.arange(1, n + 1)[None, :], axis=1)
    k = min(league.playoff_teams, n)
    made = seed <= k

    rounds = playoff_rounds(league)
    order = bracket_order(2 ** len(rounds))
    alive_slots = np.stack(
        [by_seed[:, s - 1] if s <= k else np.full(sims, -1) for s in order], axis=1
    )  # (S, size) team index or -1 for a bye
    for weeks in rounds:
        cols = [col[wk] for wk in weeks if wk in col]
        round_pts = tp[:, :, cols].sum(axis=2) if cols else np.zeros((sims, n), np.float32)
        for wk in weeks:  # playoff weeks already played count with their real scores
            if wk not in col and wk in league.results:
                real = np.array([league.results[wk].get(r, 0.0) for r in rid], np.float32)
                round_pts = round_pts + real[None, :]
        a, b = alive_slots[:, 0::2], alive_slots[:, 1::2]
        pa = np.where(a >= 0, np.take_along_axis(round_pts, np.maximum(a, 0), axis=1), -np.inf)
        pb = np.where(b >= 0, np.take_along_axis(round_pts, np.maximum(b, 0), axis=1), -np.inf)
        sa = np.where(a >= 0, np.take_along_axis(seed, np.maximum(a, 0), axis=1), 10**6)
        sb = np.where(b >= 0, np.take_along_axis(seed, np.maximum(b, 0), axis=1), 10**6)
        a_wins = (pa > pb) | ((pa == pb) & (sa < sb))  # tie → better seed
        alive_slots = np.where(a_wins, a, b)
    champ_team = alive_slots[:, 0]
    champion = np.zeros((sims, n), bool)
    has = champ_team >= 0  # no playoffs → no champion
    champion[np.arange(sims)[has], champ_team[has]] = True
    return SeasonResult(tp, made, champion, seed, wins)


def paired_delta(after: np.ndarray, before: np.ndarray) -> tuple[float, float]:
    """Mean difference and its standard error for per-sim paired values."""
    d = after.astype(np.float64) - before.astype(np.float64)
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else 0.0


def placeholder_tables() -> Tables:
    """Rough tables for tests/offline use before `ffcalc history` has run."""
    # placeholder: replaced by ffcalc.history.build_tables from Sleeper 2022–2025 history
    edges = np.array([[12, 16, 20, 24], [6, 10, 14, 18], [6, 10, 14, 18],
                      [4, 7, 10, 13], [6, 7, 8, 9], [5, 6, 7, 8]], float)  # fmt: skip
    u = np.linspace(0.005, 0.995, QGRID)
    from scipy.special import ndtri

    sd = (
        np.array([6.5, 6.0, 6.5, 5.0, 3.5, 5.0])[:, None, None]
        * np.linspace(0.8, 1.3, 5)[None, :, None]
    )
    skew = np.where(ndtri(u) > 0, 1.25, 0.8)[None, None, :]
    resid_q = sd * ndtri(u)[None, None, :] * skew
    corr_same = np.zeros((NROLES, NROLES))
    corr_opp = np.zeros((NROLES, NROLES))
    r = {name: i for i, name in enumerate(ROLES)}
    for wr in ("WR1", "WR2", "TE1"):
        corr_same[r["QB1"], r[wr]] = corr_same[r[wr], r["QB1"]] = 0.3
    corr_opp[r["QB1"], r["QB1"]] = 0.1
    corr_opp[r["QB1"], r["DEF"]] = corr_opp[r["DEF"], r["QB1"]] = -0.3
    pmf = np.array([0.4, 0.2, 0.12, 0.08, 0.06, 0.05, 0.04, 0.05])
    return Tables(
        tier_edges=edges,
        resid_q=resid_q,
        floor=np.array([-4.0, 0.0, 0.0, 0.0, -2.0, -10.0]),
        corr_same=corr_same,
        corr_opp=corr_opp,
        hazard=np.array([0.02, 0.05, 0.04, 0.04, 0.005, 0.0]),
        duration_pmf=np.tile(pmf / pmf.sum(), (NPOS, 1)),
        source="placeholder",
    )
