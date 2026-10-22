"""Plain-text rendering of a trade result (CLI)."""

from __future__ import annotations

from .engine import Context
from .trade import TeamResult, TradeResult


def _pp(x: float) -> str:
    return f"{x * 100:+.1f} pp"


def _team(ctx: Context, r: TeamResult, gives: list[str], gets: list[str]) -> list[str]:
    lines = [
        f"{r.name}",
        f"  gives: {', '.join(ctx.name(p) for p in gives) or '—'}",
        f"  gets:  {', '.join(ctx.name(p) for p in gets) or '—'}",
    ]
    if r.moves.drops:
        lines.append("  roster moves: " + ", ".join(f"drop {ctx.name(p)}" for p in r.moves.drops))
    lines += [
        f"  playoff odds: {r.playoff_before:6.1%} → {r.playoff_after:6.1%}"
        f"  ({_pp(r.playoff.value)} ± {r.playoff.se * 100:.1f})",
        f"  title odds:   {r.title_before:6.1%} → {r.title_after:6.1%}"
        f"  ({_pp(r.title.value)} ± {r.title.se * 100:.1f})",
        f"  starter points/week: {r.points_per_week.value:+.2f} ± {r.points_per_week.se:.2f}",
        f"  trade score (rest of season, playoff weeks ×2): {r.score:+.1f} pts",
    ]
    if r.decomposition:
        lines.append("  why:")
        lines += [f"    {k:<16} {v:+6.1f}" for k, v in r.decomposition.items()]
    return lines


def text(ctx: Context, res: TradeResult) -> str:
    t = res.trade
    out = [f"VERDICT for {res.a.name}: {res.verdict.upper()}", ""]
    out += _team(ctx, res.a, t.a_gives, t.b_gives) + [""]
    out += _team(ctx, res.b, t.b_gives, t.a_gives) + [""]
    if res.market is not None:
        m = res.market
        out.append(
            f"Market (FantasyCalc): you give {m.a_gives.total:,}, you get {m.b_gives.total:,}"
            f" ({m.gap:+.0%})" + ("  ← you're overpaying" if m.overpay else "")
        )
        unvalued = [
            p for side in (m.a_gives, m.b_gives) for p in side.players if not side.values.get(p)
        ]
        if unvalued:
            out.append(
                "  not in FantasyCalc's top list (counted as 0): "
                + ", ".join(ctx.name(p) for p in unvalued)
            )
    if res.acceptance is not None:
        out.append(f"Chance {res.b.name} accepts (rough heuristic): {res.acceptance:.0%}")
    out.append("")
    out.append(f"{res.sims:,} simulated seasons, seed {res.seed}; ± is one standard error.")
    out += [f"note: {w}" for w in res.warnings]
    return "\n".join(out)
