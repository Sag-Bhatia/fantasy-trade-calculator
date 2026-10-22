import { ChevronLeft, Copy, RotateCcw, Scale } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api, ApiError } from "../api";
import { useShell } from "../components/Shell";
import { Button, Card, ErrorNote, Pill, Segmented, Spinner } from "../components/ui";
import { pct, pp, signed } from "../format";
import { recall, remember, useCountUp, useLoad } from "../hooks";
import { OfferCard } from "./Finder";
import type { BalanceResult, Change, GradeResult, TeamResult } from "../types";

const EXPLAIN: Record<string, string> = {
  Starters: "Who you’d start in an average week with everyone healthy. Swapping two starters for one better one shows up here.",
  Depth: "How well the bench covers injuries over the season. Sending two players away for one thins it out.",
  "Roster cuts": "The bench player cut to make room for the extra players you receive.",
  "Schedule & byes": "Real weekly matchups and bye weeks for the players involved.",
  "Injury risk": "Current injury news, like a player listed as questionable this week.",
  "Playoff weeks": "Playoff weeks count double, because that’s when titles are won.",
};

export function Result() {
  const { league, team, base } = useShell();
  const [params] = useSearchParams();
  const partner = Number(params.get("partner"));
  const give = (params.get("give") ?? "").split(",").filter(Boolean);
  const get = (params.get("get") ?? "").split(",").filter(Boolean);
  const sims = Number(params.get("sims")) || 3000;
  const key = `grade:${league.league_id}:${team}:${partner}:${give.join(",")}:${get.join(",")}:${sims}`;
  const [result, setResult] = useState<GradeResult | undefined>(() => recall<GradeResult>(key));
  const [steps, setSteps] = useState<string[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    // the page stays mounted when only the trade in the URL changes: never show the last one's numbers
    const hit = recall<GradeResult>(key);
    setResult(hit);
    setError("");
    if (hit) return;
    let live = true;
    setSteps([]);
    api
      .grade({ league_id: league.league_id, team, partner, give, get, sims }, (m) => live && setSteps((s) => [...s, m]))
      .then((r) => {
        remember(key, r);
        if (live) setResult(r);
      })
      .catch((e) => live && setError(e instanceof ApiError ? e.message : "Something went wrong. Try again."));
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  const mine = useLoad(`roster:${league.league_id}:${team}`, () => api.roster(league.league_id, team));
  const theirs = useLoad(`roster:${league.league_id}:${partner}`, () => api.roster(league.league_id, partner));
  const nameOf = (id: string) =>
    [...(mine.data?.players ?? []), ...(theirs.data?.players ?? [])].find((p) => p.id === id)?.name ?? "";
  const deal = { give: give.map(nameOf).filter(Boolean).join(", ") || "nobody", get: get.map(nameOf).filter(Boolean).join(", ") || "nobody" };
  const editHref = `${base}?${new URLSearchParams({ partner: String(partner), give: give.join(","), get: get.join(",") })}`;
  return (
    <main className="mx-auto flex max-w-[1400px] flex-col gap-7 px-5 py-6 md:px-10">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <Link to={editHref} className="inline-flex min-h-11 items-center gap-1.5 font-bold text-you">
          <ChevronLeft size={18} strokeWidth={2.6} aria-hidden="true" /> Edit the trade
        </Link>
        <span className="text-base">
          You send <strong className="text-you-deep">{deal.give}</strong> and get <strong className="text-them-deep">{deal.get}</strong>
        </span>
      </div>
      {error && <ErrorNote>{error}</ErrorNote>}
      {!error && !result && <Simulating sims={sims} steps={steps} partnerName={league.teams.find((t) => t.roster_id === partner)?.name ?? ""} />}
      {result && <Graded key={key} r={result} base={base} deal={deal} trade={{ league_id: league.league_id, team, partner, give, get, sims }} />}
    </main>
  );
}

function Simulating({ sims, steps, partnerName }: { sims: number; steps: string[]; partnerName: string }) {
  return (
    <section aria-live="polite" className="sticker-lg grid overflow-hidden rounded-[28px] md:grid-cols-2">
      <div className="flex flex-col gap-4 bg-you p-8 text-white">
        <span className="text-lg font-bold">You</span>
        <span className="num text-7xl font-black">··</span>
      </div>
      <div className="flex flex-col items-end gap-4 bg-them p-8">
        <span className="text-lg font-bold">{partnerName}</span>
        <span className="num text-7xl font-black">··</span>
      </div>
      <div className="col-span-full flex flex-col gap-3 bg-white p-6">
        <span className="font-display text-xl font-extrabold">Playing out {sims.toLocaleString()} seasons…</span>
        <div className="h-4 overflow-hidden rounded-full border-2 border-ink bg-line">
          <div className="h-full w-1/3 animate-[slide_1.2s_ease-in-out_infinite] rounded-full bg-volt motion-reduce:animate-none" />
        </div>
        <ul className="flex flex-col gap-1 text-[15px] text-muted">
          {steps.map((s, i) => (
            <li key={i}>✓ {s}</li>
          ))}
        </ul>
        <style>{`@keyframes slide{0%{transform:translateX(-100%)}100%{transform:translateX(300%)}}`}</style>
      </div>
    </section>
  );
}

function verdictText(r: GradeResult): [string, string] {
  const b = r.b.title;
  const them = b.value > 2 * b.se ? "It helps them too" : b.value < -2 * b.se ? "It hurts them, so expect a no" : "About even for them";
  if (r.verdict === "Accept") return ["Take it.", them];
  if (r.verdict === "Decline") return ["Pass.", them];
  return ["Too close to call", them];
}

function summary(r: GradeResult): string {
  const a = r.a;
  const d = a.title.value * 100;
  const se = a.title.se * 100;
  const head =
    r.verdict === "Accept"
      ? `Your title odds go up ${d.toFixed(1)} points, well beyond the margin of error (± ${se.toFixed(1)}).`
      : r.verdict === "Decline"
        ? `Your title odds go down ${Math.abs(d).toFixed(1)} points, well beyond the margin of error (± ${se.toFixed(1)}).`
        : `Your title odds change ${signed(d)} points, inside the margin of error (± ${se.toFixed(1)}).`;
  const parts = [...a.decomposition].filter((p) => Math.abs(p.value) >= 1).sort((x, y) => Math.abs(y.value) - Math.abs(x.value));
  const words: Record<string, string> = {
    Starters: "who you’d start",
    Depth: "bench depth",
    "Roster cuts": "the roster cut",
    "Schedule & byes": "matchups and byes",
    "Injury risk": "injury news",
    "Playoff weeks": "the playoff weeks",
  };
  const reasons = parts.slice(0, 2).map((p) => `${words[p.part]} (${signed(p.value, 0)})`);
  const why = reasons.length ? ` The biggest reasons: ${reasons.join(" and ")}.` : "";
  const b = r.b;
  const them = ` ${b.name}: ${pp(b.title.value)} title odds, ${pp(b.playoff.value)} playoff odds.`;
  return head + why + them;
}

type TradeRequest = { league_id: string; team: number; partner: number; give: string[]; get: string[]; sims: number };

function Graded({ r, base, deal, trade }: { r: GradeResult; base: string; deal: { give: string; get: string }; trade: TradeRequest }) {
  const [run, setRun] = useState(0);
  const t = useCountUp(run);
  const [side, setSide] = useState<"a" | "b">("a");
  const [open, setOpen] = useState<string>("Starters");
  const [week, setWeek] = useState(r.weeks[Math.min(8, r.weeks.length - 1)] ?? r.weeks[0]);
  const [copied, setCopied] = useState(false);
  const [title, sub] = verdictText(r);
  const team = side === "a" ? r.a : r.b;

  async function copyOffer() {
    const them = r.b.title.value > 2 * r.b.title.se ? " The numbers say it helps both of us." : "";
    const text = `Trade offer for ${r.b.name}: I send ${deal.give} for ${deal.get}.${them}`;
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  }

  return (
    <>
      <section aria-label="Verdict" className="sticker-lg relative grid overflow-hidden rounded-[28px] shadow-[8px_8px_0_#12131C] md:grid-cols-2">
        <Half team={r.a} t={t} tone="you" />
        <Half team={r.b} t={t} tone="them" />
        {/* a band above the halves until there is room for the tilted sticker over the seam */}
        <div className="pointer-events-none order-first col-span-full flex flex-col items-center gap-1 border-b-[2.5px] border-ink xl:order-none bg-volt px-6 py-4 text-center xl:absolute xl:top-1/2 xl:left-1/2 xl:-translate-x-1/2 xl:-translate-y-1/2 xl:-rotate-[5deg] xl:rounded-[22px] xl:border-[2.5px] xl:shadow-[4px_4px_0_#12131C]">
          <span className="font-display text-3xl leading-none font-black xl:text-[36px]">{title}</span>
          <span className="text-sm font-bold">{sub}</span>
        </div>
      </section>

      <div className="flex flex-wrap items-center gap-4">
        <p className="min-w-[280px] flex-1 text-lg leading-relaxed">{summary(r)}</p>
        <Button onClick={() => setRun((x) => x + 1)}>
          <RotateCcw size={18} aria-hidden="true" /> Watch the seasons play out again
        </Button>
      </div>

      {!(r.a.title.value > r.a.title.se && r.b.title.value > r.b.title.se) && <MakeFair trade={trade} base={base} partnerName={r.b.name} />}

      {r.warnings.length > 0 && (
        <ul className="flex flex-col gap-1 text-[15px] text-muted">
          {r.warnings.map((w) => (
            <li key={w}>{w}</li>
          ))}
        </ul>
      )}

      <div className="flex flex-wrap items-start gap-6">
        <Card className="flex min-w-0 flex-[3_1_600px] flex-col gap-4 p-6" aria-labelledby="why">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 id="why" className="font-display text-2xl font-extrabold">
              Why
            </h2>
            <Segmented
              label="Show reasons for"
              value={side}
              onChange={setSide}
              options={[
                { value: "a", label: "You", activeClass: "bg-you text-white" },
                { value: "b", label: r.b.name, activeClass: "bg-them text-ink" },
              ]}
            />
          </div>
          <WhyBars team={team} t={t} open={open} setOpen={setOpen} />
          <p className="text-sm text-muted">
            Expected starter points over the rest of the season, added one piece at a time. Together: {signed(team.score)} points. Tap a row for what it means.
          </p>
        </Card>

        <Card className="flex min-w-0 flex-[2_1_420px] flex-col gap-4 p-6" aria-labelledby="weeks">
          <h2 id="weeks" className="font-display text-2xl font-extrabold">
            Week by week
          </h2>
          <WeekTiles r={r} week={week} setWeek={setWeek} />
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
        <Card className="flex flex-col gap-3 p-6" aria-labelledby="moves">
          <h2 id="moves" className="font-display text-xl font-extrabold">
            Roster moves
          </h2>
          <Moves r={r} />
        </Card>
        <Card className="flex flex-col gap-3 p-6" aria-labelledby="mkt">
          <h2 id="mkt" className="font-display text-xl font-extrabold">
            Market check
          </h2>
          {r.market ? (
            <>
              <div className="flex flex-wrap items-end gap-5">
                <div className="flex flex-col">
                  <span className="text-[13px] text-muted">You send</span>
                  <span className="num text-2xl font-extrabold text-you-deep">{r.market.give.toLocaleString()}</span>
                </div>
                <div className="flex flex-col">
                  <span className="text-[13px] text-muted">You get</span>
                  <span className="num text-2xl font-extrabold text-them-deep">{r.market.get.toLocaleString()}</span>
                </div>
                <Pill tone={r.market.gap >= 0 ? "up" : "down"}>
                  {signed(r.market.gap * 100, 0)}% for you
                </Pill>
              </div>
              {r.market.overpay && <p className="font-semibold text-down">By market value you’re giving up more than 15% extra.</p>}
              {r.market.unvalued.length > 0 && (
                <p className="text-sm text-muted">Not in FantasyCalc’s list (counted as 0): {r.market.unvalued.join(", ")}</p>
              )}
              <p className="text-sm text-muted">FantasyCalc values, built from real trades on fantasy sites.</p>
            </>
          ) : (
            <p className="text-muted">Market values aren’t available right now.</p>
          )}
        </Card>
        <section aria-labelledby="yes" className="sticker-lg flex flex-col gap-2 rounded-[22px] bg-ink p-6 text-white shadow-[6px_6px_0_#FF7A1F]">
          <h2 id="yes" className="font-display text-xl font-extrabold">
            Will they say yes?
          </h2>
          <span className="num text-5xl font-black text-volt">{r.acceptance == null ? "—" : `${Math.round(r.acceptance * 100)}%`}</span>
          <span className="text-sm leading-relaxed text-[#C9CCD8]">
            A rough rule of thumb, not a measurement. It rises when the deal helps their title odds and gives them more market value.
          </span>
        </section>
      </div>

      <div className="flex flex-wrap gap-3.5">
        <Button tone="volt" size="lg" onClick={copyOffer}>
          <Copy size={20} aria-hidden="true" /> {copied ? "Copied" : "Copy the offer for Sleeper chat"}
        </Button>
        <Link to={`${base}/find`} className="sticker press inline-flex min-h-[60px] items-center rounded-2xl bg-white px-6 text-lg font-bold">
          See other trades
        </Link>
      </div>
      <p className="text-sm text-muted">
        {r.sims.toLocaleString()} simulated seasons, seed {r.seed}. ± is the margin of error; differences smaller than about twice it are noise.
      </p>
    </>
  );
}

function Half({ team, t, tone }: { team: TeamResult; t: number; tone: "you" | "them" }) {
  const you = tone === "you";
  const before = team.title_before * 100;
  const delta = team.title.value * 100;
  return (
    <div
      className={`relative flex flex-col gap-3.5 overflow-hidden p-7 pb-9 md:p-9 ${you ? "bg-you text-white" : "items-end bg-them text-right text-ink"}`}
    >
      <div
        className={`pointer-events-none absolute -bottom-24 h-64 w-64 rounded-full border-[22px] ${you ? "-right-16 border-white/12" : "-left-16 border-ink/10"}`}
        aria-hidden="true"
      />
      <span className="text-lg font-bold">{you ? `You, ${team.name}` : team.name}</span>
      <div className={`flex items-baseline gap-2.5 ${you ? "" : "flex-row-reverse"}`}>
        <span className="num text-7xl leading-[0.85] font-black md:text-[112px]">{signed(delta * t)}</span>
        <span className="text-lg font-bold">title odds</span>
      </div>
      <div className="flex w-full max-w-[440px] flex-col gap-2 xl:max-w-[320px]">
        <div className="flex justify-between text-[15px] font-semibold">
          <span>{pct(team.title_before)} now</span>
          <span>{(before + delta * t).toFixed(1)}% with it</span>
        </div>
        <div className={`relative h-[22px] overflow-hidden rounded-full border-2 border-ink ${you ? "bg-white/20" : "bg-ink/15"}`}>
          <div className={`absolute inset-y-0 left-0 ${you ? "bg-white/55" : "bg-white/60"}`} style={{ width: `${Math.min(before, 100)}%` }} />
          <div
            className={`absolute inset-y-0 ${delta >= 0 ? (you ? "bg-volt" : "bg-ink") : "bg-down-bright"}`}
            style={{ left: `${Math.min(before, before + delta * t)}%`, width: `${Math.abs(delta * t)}%` }}
          />
        </div>
        <span className="text-[15px]">
          ± {(team.title.se * 100).toFixed(1)} margin of error, playoff odds {pct(team.playoff_before)} to {pct(team.playoff_after)}, {signed(team.ppw.value)} starter
          points a week
        </span>
      </div>
    </div>
  );
}

function WhyBars({ team, t, open, setOpen }: { team: TeamResult; t: number; open: string; setOpen: (v: string) => void }) {
  const lim = Math.max(10, ...team.decomposition.map((p) => Math.abs(p.value))) * 1.08;
  return (
    <div className="flex flex-col gap-2">
      {team.decomposition.map(({ part, value }) => {
        const isOpen = open === part;
        const w = (Math.abs(value) * t) / lim / 2;
        return (
          <div key={part} className={`rounded-[14px] border-2 ${isOpen ? "border-ink bg-court" : "border-transparent"}`}>
            <button
              type="button"
              aria-expanded={isOpen}
              onClick={() => setOpen(isOpen ? "" : part)}
              className="grid min-h-[52px] w-full grid-cols-[120px_minmax(0,1fr)_64px] items-center gap-3 px-3 text-left md:grid-cols-[150px_minmax(0,1fr)_76px]"
            >
              <span className="text-[15px] font-bold md:text-base">{part}</span>
              <span className="relative block h-[26px]">
                <span className="absolute -top-1 -bottom-1 left-1/2 w-0.5 bg-ink" />
                <span
                  className={`absolute top-0.5 h-[22px] rounded-md border-2 border-ink ${value >= 0 ? "bg-up-bright" : "bg-down-bright"}`}
                  style={{ left: value >= 0 ? "50%" : `${50 - w * 100}%`, width: `${Math.max(w * 100, 1)}%` }}
                />
              </span>
              <span className={`num text-right text-lg font-extrabold ${value >= 0 ? "text-up" : "text-down"}`}>{signed(value)}</span>
            </button>
            {isOpen && <p className="px-3 pb-3.5 text-[15px] leading-relaxed text-muted md:pl-[175px]">{EXPLAIN[part]}</p>}
          </div>
        );
      })}
    </div>
  );
}

function WeekTiles({ r, week, setWeek }: { r: GradeResult; week: number; setWeek: (w: number) => void }) {
  const i = r.weeks.indexOf(week);
  const tint = (v: number) =>
    v >= 3 ? "bg-up-bright" : v > 0.05 ? "bg-up-tint" : v <= -3 ? "bg-down-bright" : v < -0.05 ? "bg-down-tint" : "bg-white";
  const me = r.a.weekly_delta[i] ?? 0;
  const them = r.b.weekly_delta[i] ?? 0;
  return (
    <>
      <div className="grid grid-cols-4 gap-2 sm:grid-cols-6">
        {r.weeks.map((wk, j) => {
          const v = r.a.weekly_delta[j];
          const po = wk >= r.playoff_week_start;
          return (
            <button
              key={wk}
              type="button"
              aria-pressed={wk === week}
              aria-label={`Week ${wk}: ${signed(v)}`}
              onClick={() => setWeek(wk)}
              className={`flex min-h-16 flex-col items-center justify-center gap-0.5 rounded-[14px] border-[2.5px] border-ink ${tint(v)} ${po ? "border-dashed" : ""}`}
              style={{ boxShadow: wk === week ? "4px 4px 0 #3557FF" : "2px 2px 0 #12131C" }}
            >
              <span className="text-xs font-bold">{po ? `Playoffs ${wk}` : `Wk ${wk}`}</span>
              <span className="num text-base font-extrabold">{signed(v)}</span>
            </button>
          );
        })}
      </div>
      <div className="flex flex-col gap-1.5 rounded-2xl border-2 border-ink bg-court px-4 py-3.5">
        <span className="font-display text-lg font-extrabold">{week >= r.playoff_week_start ? `Playoff week ${week}` : `Week ${week}`}</span>
        <span className="text-[15px] leading-relaxed">
          You: {signed(me)} starter points. {r.b.name}: {signed(them)}.
          {me < -0.05 ? " A week where you’d be better off without the trade." : ""}
        </span>
      </div>
      <p className="text-sm text-muted">Change in expected starter points. Dashed tiles are the playoffs.</p>
    </>
  );
}

function Moves({ r }: { r: GradeResult }) {
  const rows = [
    ...r.a.moves.drops.map((p) => ({ k: `a-${p.id}`, tag: "Cut", tone: "down" as const, text: `You drop ${p.name}, ${p.pos}` })),
    ...r.b.moves.drops.map((p) => ({ k: `b-${p.id}`, tag: "Cut", tone: "down" as const, text: `${r.b.name} drops ${p.name}, ${p.pos}` })),
  ];
  if (!rows.length) return <p className="text-[15px]">Both rosters stay legal without any pickups or cuts.</p>;
  return (
    <ul className="flex flex-col gap-2.5">
      {rows.map((m) => (
        <li key={m.k} className="flex items-center gap-3 text-[15px]">
          <Pill tone={m.tone}>{m.tag}</Pill>
          {m.text}
        </li>
      ))}
    </ul>
  );
}

function describe(c: Change, partnerName: string): string {
  const names = (ps: { name: string }[]) => ps.map((p) => p.name).join(", ");
  if (c.give_added.length) return `You also send ${names(c.give_added)}`;
  if (c.give_removed.length) return `You keep ${names(c.give_removed)}`;
  if (c.get_added.length) return `You also get ${names(c.get_added)}`;
  return `${partnerName} keeps ${names(c.get_removed)}`;
}

/** Tries one-player changes until both teams' title odds go up. */
function MakeFair({ trade, base, partnerName }: { trade: TradeRequest; base: string; partnerName: string }) {
  const key = `balance:${trade.league_id}:${trade.team}:${trade.partner}:${trade.give.join(",")}:${trade.get.join(",")}`;
  const [found, setFound] = useState<BalanceResult | undefined>(() => recall<BalanceResult>(key));
  const [step, setStep] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function run() {
    setBusy(true);
    setError("");
    try {
      const { sims: _sims, ...body } = trade;
      const res = await api.balance(body, setStep);
      remember(key, res);
      setFound(res);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Something went wrong. Try again.");
    } finally {
      setBusy(false);
    }
  }

  const href = (o: BalanceResult["offers"][number]) =>
    `${base}/result?${new URLSearchParams({ partner: String(trade.partner), give: o.give.map((p) => p.id).join(","), get: o.get.map((p) => p.id).join(","), sims: String(trade.sims) })}`;

  return (
    <Card className="flex flex-col gap-4 p-6" aria-labelledby="fair">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex max-w-[62ch] flex-col gap-1.5">
          <h2 id="fair" className="font-display text-2xl font-extrabold">
            Make it work for both teams
          </h2>
          <p className="text-[16px] leading-relaxed">
            This trade isn’t a clear win for both sides. We’ll try adding or removing one player to find versions where both teams’ title odds go up, so {partnerName} has a reason to say yes. It takes about half a minute.
          </p>
        </div>
        {!found && (
          <Button tone="volt" size="lg" onClick={run} disabled={busy}>
            <Scale size={20} aria-hidden="true" /> {busy ? "Searching…" : "Make it fair"}
          </Button>
        )}
      </div>
      {busy && <Spinner label={step || "Starting"} />}
      {error && <ErrorNote>{error}</ErrorNote>}
      {found && found.offers.length === 0 && (
        <p className="text-[16px]">
          We tried {found.tried} one-player changes and checked the most promising ones with 10,000 seasons. None helps both teams by more than the margin of error.{" "}
          <Link to={`${base}/find`} className="font-bold text-you underline">
            Find trades
          </Link>{" "}
          searches for whole new deals instead.
        </p>
      )}
      {found && found.offers.length > 0 && (
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2 xl:grid-cols-3">
          {found.offers.map((o, i) => (
            <OfferCard key={i} o={o} base={base} to={href(o)} cta="Grade this version" note={describe(o.change, partnerName)} />
          ))}
        </div>
      )}
    </Card>
  );
}
