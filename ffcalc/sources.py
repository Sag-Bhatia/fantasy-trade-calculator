"""Bundle of data clients so the engine can swap in fakes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ffcalc.clients.espn import EspnClient
from ffcalc.clients.fantasycalc import FantasyCalcClient
from ffcalc.clients.nflverse import NflverseClient
from ffcalc.clients.sleeper import SleeperClient


@dataclass
class Sources:
    sleeper: Any
    fantasycalc: Any
    espn: Any = None
    nflverse: Any = None


def live() -> Sources:
    return Sources(
        sleeper=SleeperClient(),
        fantasycalc=FantasyCalcClient(),
        espn=EspnClient(),
        nflverse=NflverseClient(),
    )
