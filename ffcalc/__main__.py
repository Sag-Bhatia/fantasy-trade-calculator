"""Command line: `uv run python -m ffcalc <command> ...`."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ffcalc", description="Fantasy football trade calculator")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ev = sub.add_parser("evaluate", help="grade a trade")
    ev.add_argument("--league", required=True, help="Sleeper league ID")
    ev.add_argument("--team", required=True, type=int, help="your roster_id")
    ev.add_argument("--give", required=True, help="comma-separated player names or Sleeper IDs")
    ev.add_argument("--get", required=True, help="comma-separated player names or Sleeper IDs")
    ev.add_argument("--sims", type=int, default=10_000)
    ev.add_argument("--seed", type=int, default=7)

    fd = sub.add_parser("find", help="search for trades that help both teams")
    fd.add_argument("--league", required=True)
    fd.add_argument("--team", required=True, type=int)

    hi = sub.add_parser("history", help="build variance, correlation and injury tables")
    hi.add_argument("--seasons", default="2022,2023,2024,2025")

    args = ap.parse_args(argv)
    print(f"'{args.cmd}' is not built yet", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
