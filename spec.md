# Fantasy Football Trade Calculator — Spec

A local app that grades a fantasy football trade by simulating the rest of the season
for *your* league — your rosters, scoring, schedule, playoff format and waiver wire —
with and without the trade, and reports the change in playoff odds, title odds and
expected points for both teams, plus *why*.

## 1. Scope

**In (v1):** Sleeper leagues, redraft, season-long. Any roster slots Sleeper supports
(QB/RB/WR/TE/K/DEF/FLEX/SUPER_FLEX/REC_FLEX/WRRB_FLEX, bench, IR). Any Sleeper scoring.
Median-win leagues. Multi-week fantasy playoffs and two-week matchups.

**Out (v1):** ESPN/Yahoo league import, dynasty/keepers/draft picks, FAAB bidding
strategy, IDP scoring, best-ball. Don't build these.

## 2. Data sources (all free, checked 2026-10-21)

| Source | Use | Access |
|---|---|---|
| Sleeper API `api.sleeper.app/v1` | league settings, scoring, rosters, users, matchups (schedule), transactions/trades, `state/nfl`, players dump | public, no key. Players dump (~5 MB) at most once/day. Stay well under 1000 req/min. |
| Sleeper projections `api.sleeper.app/projections/nfl/{season}/{week}?season_type=regular&position[]=WR` | weekly projected **stat lines** (+ pts_std/half/ppr) | unofficial. Past seasons work (2024 verified). |
| ESPN `lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{s}/segments/0/leaguedefaults/3?view=kona_player_info&scoringPeriodId={w}` + `X-Fantasy-Filter` header | weekly projected stat lines (`statSourceId=1` = projection, `0` = actual) | unofficial. Past seasons work (2024 verified). ≤1 req/s. |
| FantasyCalc `api.fantasycalc.com/values/current?isDynasty=false&numQbs=&numTeams=&ppr=` | market trade values from real trades, `trend30Day`; carries `sleeperId`, `espnId` | public, no key. Cache 6 h. No history → not backtested. |
| nflverse via `nflreadpy` | weekly stats, play-by-play, snaps, injuries, schedules (with `spread_line`, `total_line`), `load_ff_playerids()` | free. |
| FantasyPros CSV (optional) | consensus projections the user downloads manually and drops in `data/imports/` | we never scrape FantasyPros. |

numberFire no longer exists. Every client has a fake twin that reads `tests/fixtures/`.
If an endpoint is down or changed: build against the fixture, log it in NOTES.md, move on.

### Player identity (one place)
All sources join through one ID table from `nflreadpy.load_ff_playerids()` (sleeper_id,
espn_id, fantasypros_id, gsis_id, name, position, team). Never join on names, except
FantasyPros CSV rows, which are matched on normalized name + team + position against
that table; unmatched rows are reported, not guessed.

### Caching and snapshots
`data/cache/` holds parquet/JSON with per-source TTLs (players 24 h, projections 1 h,
league/rosters 10 min, FantasyCalc 6 h, nflverse 24 h). Every fetch of weekly projections
is also written to `data/snapshots/{source}/{season}/{week}.parquet` so future backtests
have projections exactly as they were. `data/` is gitignored.

## 3. Scoring
Projections are blended as **stat lines**, then scored with the league's own
`scoring_settings` by one function in `scoring.py` (`points(stats, scoring_settings)`).
Never use a source's pre-computed `pts_ppr` etc. for valuation (fine for sanity tests).
This makes TE premium, 6-pt pass TD, bonuses, first downs etc. correct.

## 4. Projection engine

**Blend.** Per player-week, weighted mean of available sources' stat lines: Sleeper,
ESPN, in-house model, FantasyPros CSV if present. Weights per position = inverse of
backtested MAE (from `backtest.py`); equal weights until a backtest has run. Missing
source → renormalize over the rest.

**In-house model.** Trained on nflverse: trailing target share, carry share, snap share,
air yards, red-zone usage, team implied total (`(total_line ∓ spread_line)/2`), opponent
points allowed to position. Predicts stat components per position. Simple and
regularized (ridge/GBM from numpy or scikit-learn if needed); trained only on data
before the week it predicts.

**Weekly adjustments.**
- Bye week → 0, not playing.
- Opponent: multiply by shrunk opponent-vs-position factor and implied-total ratio
  (skip for a source that already adjusts; log the choice).
- Injury status → `P(plays)` and a mean multiplier given playing, by status
  (Questionable, Doubtful, Out, IR, PUP) and position, estimated from nflverse
  injuries joined to weekly stats. Out/IR → 0.

**Distributions.** Each player-week gets a mean and a skewed spread. From history,
compute residuals `actual − projection` grouped by position × projection tier
(e.g. tier edges at projection quantiles). Sampling = projection + residual drawn by
quantile from that bucket's empirical distribution, floored at the position's minimum
realistic score. Not-playing is a separate draw (zero-inflation), not part of the residual.

**Correlation.** Players in the same NFL game are sampled jointly via a Gaussian
copula: correlated normals → uniforms → empirical residual quantiles. Correlation by
position pair, same team vs opponent (e.g. QB–own WR, RB–own DEF, QB–opposing QB),
estimated from nflverse history of residuals. Players in different games independent.

## 5. Lineups and replacement level

`lineup.py`: given a roster, slot list and expected points, return the optimal legal
starting lineup. Must equal a brute-force search on all test cases (tests compare both
on random rosters). Managers set lineups on **expected** points, never on realized
outcomes; inside a sim, a player knocked out by injury is replaced by re-optimizing.

**Replacement level = the actual best free agent in this league.** Free agents = all
active NFL players − every rostered player. When a trade leaves a team with an open
roster spot (2-for-1), the spot stays open: a waiver pickup isn't part of the trade, since
the team could make it without trading. When a team receives more players than it sends,
it drops its lowest-value bench player. FAs are also used weekly to fill a slot that
would otherwise be empty (byes/injuries) — one pickup per team per week, best
available, assumed obtainable (shortcut: ignores waiver priority/FAAB).

## 6. Monte Carlo season simulator (`simulate.py`)

- Default 10,000 sims. Target: **12-team league, 10k sims, < 20 s on a laptop.**
  Fully vectorized NumPy over `(sims, weeks, players)`; only players on rosters plus the
  top ~10 FAs per position are simulated. float32.
- Remaining regular-season schedule from Sleeper matchups (`matchup_id` per week).
  Already-played weeks use real results.
- Median win if `league_average_match` is set.
- Standings: wins, then the league's tiebreaker (Sleeper default: points for).
- Playoffs from settings: `playoff_teams`, `playoff_week_start`, byes for top seeds,
  `playoff_round_type` (one-week rounds, two-week championship, or two-week rounds;
  two-week matchups sum both weeks). Consolation brackets ignored.
- Injuries: each week each player may get hurt at a position-specific weekly rate
  with a games-missed duration drawn from history (nflverse injuries/weekly).
- **Common random numbers:** draws are indexed by `(sim, week, player)` from one seeded
  generator, so a player's draws are identical no matter which team owns him. The
  with-trade and without-trade runs use the same seed and the same draws; only rosters
  differ. Report paired standard errors for every delta.
- Outputs per team: playoff %, title %, expected points/week (starters), seed
  distribution, weekly score distribution.

## 7. Trade evaluation (`trade.py`)

Input: league, the two teams, players each side gives. Runs the sim twice (CRN) and
reports for **both** teams: Δ playoff %, Δ title %, Δ expected starter points/week,
each with ± standard error.

**Playoff weighting.** The headline "trade score" weights expected-point changes in
fantasy playoff weeks 2× regular-season weeks (constant in code, documented here).
Title % already captures this naturally; the weighted score is the simple summary.

**Decomposition — why it helps or hurts.** Expected-points based, computed by
sequential ablation in this fixed order (each step's Δ vs the previous):
1. **Starters** — Δ in optimal lineup points, healthy, no byes, no FA fill.
2. **Depth** — add bench coverage of others' absences.
3. **Roster cuts** — add the bench players a team must cut to fit what it receives.
4. **Schedule and byes** — add real byes and opponent adjustments.
5. **Injury risk** — add injury status and in-season injury rates.
6. **Playoff weeks** — extra value from playoff-week weighting.
Order-dependence is stated in the UI. Components must sum to the total.

**Verdict.** From Δ title % and its SE: "Accept" if Δ > +0.5 pp and > 2 SE; "Decline"
if Δ < −0.5 pp and < −2 SE; otherwise "Roughly even". Also shows market check (§8).

**CLI.** `uv run python -m ffcalc evaluate --league ID --team ROSTER_ID --give "A,B" --get "C" [--sims N]`
prints the same numbers and decomposition.

## 8. Market values and acceptance odds (`market.py`)

- FantasyCalc values with params matched to the league (team count, QBs, PPR level).
  Report each side's total value and the gap; flag when the user overpays vs market by
  > 15%.
- Live league trades: recent `type == "trade"` transactions from the league, shown
  with their market values, so the user sees what this league actually pays.
- **Acceptance odds:** a heuristic logistic of the partner's Δ title % and the market
  value gap. Labeled "heuristic" in the UI — there's no data on rejected trades, so it
  cannot be calibrated. Never presented as a probability we measured.

## 9. Trade finder (`finder.py`)

Search my team vs each other team for 1-for-1, 2-for-1 and 2-for-2 trades that **raise
both teams' title odds**. Prune candidates by market-value band (±30%) and positional
need. Screen survivors at 1,000 sims (CRN), re-run the top 10 at 10,000, return those
with both Δ title % > 0 beyond 1 SE, ranked by the smaller of the two gains.
Budget: < 2 min for a 12-team league.

## 10. UI (React + FastAPI: `ffcalc/api.py`, `web/`)

`uv run python -m ffcalc serve` starts one local server (127.0.0.1:8000) that serves the
HTTP API (`docs/api.md`) and the built React app (`web/dist`, committed). Screens:

1. **League** — paste Sleeper league ID (or username → pick league), pick my team.
   Shows format, scoring summary, roster slots, current standings.
2. **Trade** — pick partner and players; Evaluate. Shows both teams' deltas, verdict,
   decomposition bar, weekly score distributions, market check, acceptance heuristic.
3. **Trade finder** — runs §9, lists suggestions.
4. **Accuracy** — renders REPORT.md.
5. **Welcome** — league ID or username lookup, then pick your team (remembered locally).
The server caches each league's prepared context for 10 minutes (one build at a time per
league); grades and finder runs stream progress as NDJSON. Optional FantasyPros CSV upload
on the League page (kept in memory until the server stops). Originally Streamlit; rebuilt
on the user's request (NOTES.md).

## 11. Backtesting and calibration (`backtest.py` → REPORT.md)

`uv run python -m ffcalc.backtest` generates REPORT.md from real data only. Never
hand-edit or invent numbers. Every number states its sample size.

1. **Projection accuracy.** Past seasons (2021+ where available): MAE per position
   for Sleeper, ESPN, in-house, blend, and naive baselines (season-to-date average,
   last-3-week average). Uses snapshots when present, else historical endpoint pulls.
   If a source's history is unavailable, say so and compare what exists.
2. **Distribution calibration.** Coverage of the 10/50/90% predicted quantiles.
3. **Simulator calibration.** Mid-season snapshots (weeks 6, 9, 12) → predicted
   playoff % vs actual outcome; Brier score and a reliability table (bins of 10%).
   Leagues: the user's own league history (Sleeper `previous_league_id` chain), with
   rosters at week N rebuilt by undoing later transactions. Fallback when no history:
   synthetic 12-team leagues drafted by ADP from real past seasons, scored with real
   weekly results. Report says which was used.
4. **Trade replay.** For each past trade in the user's league history: which side the
   model favored at the time vs which side gained more realized rest-of-season starter
   points. Report hit rate and n (small n stated plainly).

## 12. Layout

```
ffcalc/api.py             HTTP API (FastAPI) + serves web/dist
web/                      React + Vite + Tailwind app; web/dist is the committed build
ffcalc/__main__.py        CLI (argparse)
ffcalc/cache.py           TTL disk cache + snapshots
ffcalc/clients/           sleeper.py espn.py fantasycalc.py nflverse.py fantasypros_csv.py
ffcalc/ids.py             player ID table
ffcalc/scoring.py         stat line → points for league scoring
ffcalc/history.py         residual, correlation, injury tables from nflverse
ffcalc/projections.py     blend, adjustments, distributions, in-house model
ffcalc/lineup.py          optimal lineup, FA replacement
ffcalc/simulate.py        Monte Carlo season + playoffs
ffcalc/trade.py           evaluate, decomposition, verdict
ffcalc/market.py          FantasyCalc, league trades, acceptance heuristic
ffcalc/finder.py          trade finder
ffcalc/backtest.py        accuracy, calibration, REPORT.md
tests/, tests/fixtures/, tests/fakes.py
```

## 13. Acceptance criteria

- `uv run pytest` passes fully offline.
- Lineup optimizer matches brute force on ≥ 500 random rosters.
- Same seed → identical results; with-trade vs without-trade on a no-op trade gives
  exactly zero delta.
- 12-team, 10k-sim run < 20 s (timed test, marked `slow`).
- Decomposition components sum to the total within float tolerance.
- Two-week playoff rounds and median-win leagues covered by tests.
- REPORT.md is produced by the harness, with sample sizes, from real data.
