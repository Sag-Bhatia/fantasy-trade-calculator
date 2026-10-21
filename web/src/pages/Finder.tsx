import { ArrowRight, Search } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import { api, ApiError } from "../api";
import { useShell } from "../components/Shell";
import { Button, ErrorNote, Jersey, Segmented, Spinner } from "../components/ui";
import { signed } from "../format";
import { recall, remember } from "../hooks";
import type { FindResult, Offer } from "../types";

export function Finder() {
  const { league, team, base } = useShell();
  const key = `find:${league.league_id}:${team}`;
  const [result, setResult] = useState<FindResult | undefined>(() => recall<FindResult>(key));
  const [step, setStep] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [pos, setPos] = useState("All");
  const [sort, setSort] = useState<"both" | "me">("both");

  async function search() {
    setBusy(true);
    setError("");
    try {
      const r = await api.find({ league_id: league.league_id, team }, setStep);
      remember(key, r);
      setResult(r);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Something went wrong. Try again.");
    } finally {
      setBusy(false);
    }
  }

  const positions = ["All", ...new Set(result?.offers.flatMap((o) => o.get.map((p) => p.pos)) ?? [])];
  const offers = [...(result?.offers ?? [])]
    .filter((o) => pos === "All" || o.get.some((p) => p.pos === pos))
    .sort((a, b) => (sort === "me" ? b.me.value - a.me.value : Math.min(b.me.value, b.them.value) - Math.min(a.me.value, a.them.value)));

  return (
    <main className="mx-auto flex max-w-[1400px] flex-col gap-6 px-5 py-7 md:px-10">
      <section className="sticker-lg relative flex flex-wrap items-end justify-between gap-6 overflow-hidden rounded-[28px] bg-ink p-7 text-white shadow-[8px_8px_0_#3557FF] md:p-9">
        <div className="pointer-events-none absolute hidden md:block -top-16 right-16 h-56 w-56 rounded-full bg-them" aria-hidden="true" />
        <div className="pointer-events-none absolute hidden md:block top-16 -right-10 h-40 w-40 rounded-full bg-you" aria-hidden="true" />
        <div className="relative flex max-w-[760px] flex-col gap-3">
          <h1 className="font-display text-4xl leading-[0.98] font-black tracking-tight md:text-[54px]">Trades both of you win</h1>
          <p className="text-lg leading-relaxed text-[#D6D8E3]">
            {result
              ? `We checked ${result.checked.toLocaleString()} deals across ${result.rosters} rosters. These ${result.offers.length} raise your title odds and theirs by more than the margin of error, so the other manager has a reason to say yes.`
              : "We search every other roster for one- and two-player deals that raise your title odds and theirs, so the other manager has a reason to say yes. It takes a minute or two."}
          </p>
        </div>
        <Button tone="volt" className="relative shadow-[4px_4px_0_#fff]" onClick={search} disabled={busy}>
          <Search size={18} aria-hidden="true" /> {result ? "Search again" : "Find trades for me"}
        </Button>
      </section>

      {busy && <Spinner label={step || "Starting the search"} />}
      {error && <ErrorNote>{error}</ErrorNote>}

      {result && (
        <>
          <div className="flex flex-wrap items-center justify-between gap-3.5">
            <div role="group" aria-label="Position you get" className="flex flex-wrap gap-2">
              {positions.map((p) => (
                <button
                  key={p}
                  type="button"
                  aria-pressed={pos === p}
                  onClick={() => setPos(p)}
                  className={`min-h-11 rounded-full border-[2.5px] border-ink px-4 text-[15px] font-bold ${pos === p ? "bg-ink text-volt" : "press bg-white shadow-[3px_3px_0_#12131C]"}`}
                >
                  {p === "All" ? "Any position" : `Get a ${p}`}
                </button>
              ))}
            </div>
            <Segmented
              label="Sort"
              value={sort}
              onChange={setSort}
              options={[
                { value: "both", label: "Best for both" },
                { value: "me", label: "Best for me", activeClass: "bg-you text-white" },
              ]}
            />
          </div>
          {offers.length === 0 && (
            <p className="text-lg">
              {result.offers.length === 0
                ? "No trade clearly helps both teams right now. That’s common when rosters are balanced; try again after injuries or waiver moves."
                : "No offers bring you that position. Try another filter."}
            </p>
          )}
          <div className="grid grid-cols-1 gap-6 md:grid-cols-[repeat(auto-fill,minmax(400px,1fr))]">
            {offers.map((o, i) => (
              <OfferCard key={i} o={o} base={base} />
            ))}
          </div>
          <p className="text-sm text-muted">From 10,000 simulated seasons. At most two offers per team, and only the best version of each deal.</p>
        </>
      )}
    </main>
  );
}

/** One suggested trade. `to`/`cta` change where the button goes; `note` says what's different. */
export function OfferCard({ o, base, to, cta = "Grade this trade in detail", note }: { o: Offer; base: string; to?: string; cta?: string; note?: string }) {
  const q = new URLSearchParams({ partner: String(o.partner.roster_id), give: o.give.map((p) => p.id).join(","), get: o.get.map((p) => p.id).join(",") });
  const gap = o.market_gap;
  return (
    <article className="sticker-lg flex flex-col overflow-hidden rounded-[22px] bg-white">
      <Jersey team={o.get[0]?.nfl_team ?? null} className="h-3.5 border-b-[2.5px] border-ink" />
      <header className="flex items-center justify-between gap-3 px-5 pt-3.5 pb-1">
        <span className="font-bold">
          with {o.partner.name} <span className="font-medium text-muted">{o.partner.record}</span>
        </span>
        {gap != null && (
          <span className={`num rounded-full border-2 border-ink px-2.5 py-1 text-[13px] font-extrabold ${gap >= 0 ? "bg-up-tint text-[#065C34]" : "bg-down-tint text-[#A3122F]"}`}>
            Market {signed(gap * 100, 0)}%
          </span>
        )}
      </header>
      {note && <p className="px-5 pt-1 text-[15px] font-bold">{note}</p>}
      <div className="grid grid-cols-2 gap-2.5 px-5 py-2.5">
        <div className="flex flex-col gap-1 rounded-[14px] bg-you-tint px-3 py-2.5">
          <span className="text-xs font-bold text-you-deep">You send</span>
          <span className="text-[15px] leading-snug font-bold">{o.give.map((p) => p.name).join(", ")}</span>
        </div>
        <div className="flex flex-col gap-1 rounded-[14px] bg-them-tint px-3 py-2.5">
          <span className="text-xs font-bold text-them-deep">You get</span>
          <span className="text-[15px] leading-snug font-bold">{o.get.map((p) => p.name).join(", ")}</span>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-2.5 px-5 pt-1 pb-4">
        <div className="rounded-[14px] border-2 border-ink bg-you px-3 py-2.5 text-white">
          <span className="text-xs font-bold">Your title odds</span>
          <div className="num text-[34px] leading-none font-black">{signed(o.me.value * 100)}</div>
          <span className="text-xs">± {(o.me.se * 100).toFixed(1)}</span>
        </div>
        <div className="rounded-[14px] border-2 border-ink bg-them px-3 py-2.5">
          <span className="text-xs font-bold">Their title odds</span>
          <div className="num text-[34px] leading-none font-black">{signed(o.them.value * 100)}</div>
          <span className="text-xs">± {(o.them.se * 100).toFixed(1)}</span>
        </div>
      </div>
      <Link
        to={to ?? `${base}?${q}`}
        className="mt-auto flex min-h-[54px] items-center justify-center gap-2 border-t-[2.5px] border-ink bg-volt font-bold hover:bg-[#c3f21f]"
      >
        {cta} <ArrowRight size={18} strokeWidth={2.6} aria-hidden="true" />
      </Link>
    </article>
  );
}
