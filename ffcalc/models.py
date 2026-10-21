"""Plain data types shared by every module."""

from __future__ import annotations

from dataclasses import dataclass, field

POSITIONS = ("QB", "RB", "WR", "TE", "K", "DEF")

# Sleeper roster slot → positions that may fill it.
SLOT_ELIGIBILITY: dict[str, tuple[str, ...]] = {
    "QB": ("QB",),
    "RB": ("RB",),
    "WR": ("WR",),
    "TE": ("TE",),
    "K": ("K",),
    "DEF": ("DEF",),
    "FLEX": ("RB", "WR", "TE"),
    "WRRB_FLEX": ("RB", "WR"),
    "REC_FLEX": ("WR", "TE"),
    "SUPER_FLEX": ("QB", "RB", "WR", "TE"),
}
NON_STARTING = {"BN", "IR", "TAXI"}


@dataclass
class Team:
    roster_id: int
    owner_id: str | None
    name: str
    players: list[str]  # every player on the roster (Sleeper ids), incl. reserve
    reserve: list[str] = field(default_factory=list)  # IR slots
    wins: int = 0
    losses: int = 0
    ties: int = 0
    points_for: float = 0.0
    points_against: float = 0.0


@dataclass
class League:
    league_id: str
    name: str
    season: int
    num_teams: int
    slots: list[str]  # starting slots only, e.g. ["QB","RB","RB","WR","WR","TE","FLEX","K","DEF"]
    bench_size: int
    ir_slots: int
    scoring: dict[str, float]
    playoff_teams: int
    playoff_week_start: int
    playoff_round_type: int  # 0 one-week rounds, 1 two-week final, 2 two-week rounds
    median_win: bool
    trade_deadline: int  # last week trades are allowed (0 = none)
    current_week: int  # first week not yet fully scored
    teams: list[Team]
    # week → list of (roster_id, roster_id) pairs, regular season only
    schedule: dict[int, list[tuple[int, int]]]
    # week → roster_id → points, for weeks already scored
    results: dict[int, dict[int, float]]
    previous_league_id: str | None = None

    @property
    def roster_size(self) -> int:
        return len(self.slots) + self.bench_size

    @property
    def num_qbs(self) -> int:
        return sum(s in ("QB", "SUPER_FLEX") for s in self.slots)

    @property
    def last_regular_week(self) -> int:
        return self.playoff_week_start - 1

    def team(self, roster_id: int) -> Team:
        return next(t for t in self.teams if t.roster_id == roster_id)
