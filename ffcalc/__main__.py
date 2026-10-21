"""Command line: `uv run python -m ffcalc <command> ...`."""

from __future__ import annotations

import argparse
import logging
import sys

import polars as pl


def _resolve(ctx, query: str, owners: set[int] | None = None) -> str:  # noqa: ANN001
    """User-typed player name or Sleeper id → Sleeper id, among rostered players."""
    from ffcalc import ids

    rostered = {
        p for t in ctx.league.teams for p in t.players if owners is None or t.roster_id in owners
    }
    table = ctx.table.filter(pl.col("sleeper_id").is_in(list(rostered)))
    pid = ids.lookup(table, query.strip())
    if pid is None:
        q = ids.normalize_name(query)
        close = table.filter(pl.col("search_name").str.starts_with(q[:4]))["name"].to_list()[:5]
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise SystemExit(f"Couldn't find exactly one rostered player matching '{query}'.{hint}")
    return pid


def _owner(ctx, pid: str) -> int:  # noqa: ANN001
    return next(t.roster_id for t in ctx.league.teams if pid in t.players)


def evaluate(args: argparse.Namespace) -> int:
    from ffcalc import engine, report
    from ffcalc.trade import Trade, TradeError
    from ffcalc.trade import evaluate as run

    ctx = engine.prepare(args.league, progress=lambda m: print(f"… {m}", file=sys.stderr))
    if args.team not in {t.roster_id for t in ctx.league.teams}:
        hint = f"run `python -m ffcalc teams --league {args.league}`"
        raise SystemExit(f"No team {args.team} in this league; {hint}.")
    give = [_resolve(ctx, q, {args.team}) for q in args.give.split(",") if q.strip()]
    get = [_resolve(ctx, q) for q in args.get.split(",") if q.strip()]
    partners = {_owner(ctx, p) for p in get}
    if len(partners) != 1 or args.team in partners:
        raise SystemExit("Everything in --get must come from one other team.")
    try:
        result = run(
            ctx, Trade(args.team, partners.pop(), give, get), sims=args.sims, seed=args.seed
        )
    except TradeError as e:
        raise SystemExit(str(e)) from e
    print(report.text(ctx, result))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.ERROR)
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

    tm = sub.add_parser("teams", help="list a league's teams and roster ids")
    tm.add_argument("--league", required=True)

    hi = sub.add_parser("history", help="build variance, correlation and injury tables")
    hi.add_argument("--league", required=True, help="tables use this league's scoring")

    args = ap.parse_args(argv)
    if args.cmd == "evaluate":
        return evaluate(args)
    if args.cmd == "find":
        from ffcalc import engine
        from ffcalc.finder import find_trades

        ctx = engine.prepare(args.league, progress=lambda m: print(f"… {m}", file=sys.stderr))
        found = find_trades(ctx, args.team, progress=lambda m: print(f"… {m}", file=sys.stderr))
        if not found:
            print("No trade clearly raises both teams' title odds right now.")
        for s in found:
            partner = ctx.league.team(s.trade.team_b).name
            print(
                f"{partner}: give {', '.join(ctx.name(p) for p in s.trade.a_gives)}"
                f" for {', '.join(ctx.name(p) for p in s.trade.b_gives)}"
                f"  | your title odds {s.a_title.value * 100:+.1f} ± {s.a_title.se * 100:.1f} pp,"
                f" theirs {s.b_title.value * 100:+.1f} ± {s.b_title.se * 100:.1f} pp"
            )
        return 0
    if args.cmd == "teams":
        from ffcalc.league import load_league
        from ffcalc.sources import live

        lg = load_league(args.league, live().sleeper)
        for t in lg.teams:
            print(f"{t.roster_id:>3}  {t.name}  ({t.wins}-{t.losses})")
        return 0
    if args.cmd == "history":
        from ffcalc import engine

        ctx = engine.prepare(args.league, progress=lambda m: print(f"… {m}", file=sys.stderr))
        print(f"tables: {ctx.tables.source}")
        return 0
    print(f"'{args.cmd}' is not built yet", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
