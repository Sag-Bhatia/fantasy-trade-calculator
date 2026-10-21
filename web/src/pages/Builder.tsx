import { ArrowRight } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { api } from "../api";
import { PlayerCard } from "../components/PlayerCard";
import { useShell } from "../components/Shell";
import { Button, ErrorNote, Spinner } from "../components/ui";
import { signed } from "../format";
import { useLoad } from "../hooks";
import type { Preview, RosterPlayer } from "../types";

const ids = (v: string | null) => (v ? v.split(",").filter(Boolean) : []);

export function Builder() {
  const { league, team, base } = useShell();
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const others = league.teams.filter((t) => t.roster_id !== team);
  const initialPartner = Number(params.get("partner")) || others[0]?.roster_id;
  const [partner, setPartner] = useState<number>(initialPartner);
  const [give, setGive] = useState<string[]>(ids(params.get("give")));
  const [get, setGet] = useState<string[]>(ids(params.get("get")));
  const [sims, setSims] = useState(3000);
  const [preview, setPreview] = useState<Preview | null>(null);

  const mine = useLoad(`roster:${league.league_id}:${team}`, () => api.roster(league.league_id, team));
  const theirs = useLoad(partner ? `roster:${league.league_id}:${partner}` : null, () => api.roster(league.league_id, partner));
  const partnerName = league.teams.find((t) => t.roster_id === partner)?.name ?? "them";

  useEffect(() => {
    if (!give.length && !get.length) {
      setPreview(null);
      return;
    }
    let live = true;
    const id = setTimeout(() => {
      api
        .preview({ league_id: league.league_id, team, partner, give, get })
        .then((p) => live && setPreview(p))
        .catch(() => live && setPreview(null));
    }, 200);
    return () => {
      live = false;
      clearTimeout(id);
    };
  }, [give, get, partner, team, league.league_id]);

  const byId = useMemo(() => {
    const m = new Map<string, RosterPlayer>();
    [...(mine.data?.players ?? []), ...(theirs.data?.players ?? [])].forEach((p) => m.set(p.id, p));
    return m;
  }, [mine.data, theirs.data]);

  const toggle = (list: string[], set: (v: string[]) => void, id: string) =>
    set(list.includes(id) ? list.filter((x) => x !== id) : [...list, id]);

  function grade() {
    const q = new URLSearchParams({ partner: String(partner), give: give.join(","), get: get.join(","), sims: String(sims) });
    navigate(`${base}/result?${q}`);
  }

  return (
    <main className="mx-auto flex max-w-[1400px] flex-col gap-6 px-5 pt-7 pb-24 md:px-10 lg:pb-7">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="flex flex-col gap-2">
          <h1 className="font-display text-4xl leading-[0.95] font-black tracking-tight md:text-[56px]">Build a trade</h1>
          <p className="text-[17px] text-muted">Tap players on either side. Numbers are projected points this week.</p>
        </div>
        <label className="flex items-center gap-2.5 font-semibold">
          <span>Trade with</span>
          <select
            value={partner}
            onChange={(e) => {
              setPartner(Number(e.target.value));
              setGet([]);
            }}
            className="sticker min-h-12 rounded-xl bg-white px-3.5 font-bold"
          >
            {others.map((t) => (
              <option key={t.roster_id} value={t.roster_id}>
                {t.name} ({t.wins}-{t.losses})
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="flex flex-wrap items-start gap-6">
        <RosterColumn
          title="Your roster"
          side="you"
          state={mine}
          selected={give}
          onToggle={(id) => toggle(give, setGive, id)}
        />

        <section aria-labelledby="deal" className="flex w-full min-w-[300px] flex-[0_1_380px] flex-col gap-4 lg:sticky lg:top-4">
          <div className="sticker-lg overflow-hidden rounded-[20px] bg-white">
            <div className="flex items-center justify-between border-b-[2.5px] border-ink px-5 py-4">
              <h2 id="deal" className="font-display text-xl font-extrabold">
                The deal
              </h2>
              <span className="num font-bold">{preview?.shape ?? `${give.length} for ${get.length}`}</span>
            </div>
            <SlipSide label="You send" tone="you" players={give.map((id) => byId.get(id)).filter(Boolean) as RosterPlayer[]} empty="Tap your players to add them." />
            <SlipSide
              label="You get"
              tone="them"
              players={get.map((id) => byId.get(id)).filter(Boolean) as RosterPlayer[]}
              empty={`Tap ${partnerName}’s players to add them.`}
            />
            <div className="flex flex-col gap-3 border-t-[2.5px] border-ink px-5 py-4">
              <span className="text-sm font-bold">Quick read: this week’s starting lineup</span>
              <div className="grid grid-cols-2 gap-2.5">
                <div className="rounded-[14px] border-2 border-ink bg-you px-3 py-2.5 text-white">
                  <span className="text-[13px] font-semibold">You</span>
                  <div className="num text-[30px] font-black">{preview ? signed(preview.me) : "0.0"}</div>
                </div>
                <div className="rounded-[14px] border-2 border-ink bg-them px-3 py-2.5">
                  <span className="block truncate text-[13px] font-semibold">{partnerName}</span>
                  <div className="num text-[30px] font-black">{preview ? signed(preview.them) : "0.0"}</div>
                </div>
              </div>
              <p className="text-sm leading-relaxed text-muted">
                {preview?.note ?? "Pick players on both sides to see how each lineup changes. The full grade plays out the rest of the season."}
              </p>
            </div>
          </div>

          <fieldset>
            <legend className="mb-2 text-sm font-bold">How precise</legend>
            <div className="grid grid-cols-2 gap-2.5">
              {[
                { v: 3000, label: "Quick", sub: "3,000 seasons, ~5 s" },
                { v: 10000, label: "Full", sub: "10,000 seasons, ~15 s" },
              ].map((o) => (
                <button
                  key={o.v}
                  type="button"
                  aria-pressed={sims === o.v}
                  onClick={() => setSims(o.v)}
                  className={`flex min-h-14 flex-col items-center justify-center rounded-[14px] border-[2.5px] border-ink ${
                    sims === o.v ? "bg-ink text-volt" : "press bg-white shadow-[3px_3px_0_#12131C]"
                  }`}
                >
                  <span className="font-bold">{o.label}</span>
                  <span className="text-xs">{o.sub}</span>
                </button>
              ))}
            </div>
          </fieldset>

          <Button tone="volt" size="lg" onClick={grade} disabled={!give.length && !get.length}>
            Grade this trade <ArrowRight size={22} strokeWidth={2.6} aria-hidden="true" />
          </Button>
        </section>

        <RosterColumn
          title={`${partnerName}’s roster`}
          side="them"
          state={theirs}
          selected={get}
          onToggle={(id) => toggle(get, setGet, id)}
        />
      </div>

      {/* stacked layout: the deal slip is out of view while you pick their players */}
      {(give.length > 0 || get.length > 0) && (
        <div className="sticker fixed inset-x-3 bottom-[84px] z-10 flex items-center justify-between gap-3 rounded-2xl bg-white py-2 pr-2 pl-4 md:bottom-4 lg:hidden">
          <span className="text-[15px] font-bold">
            You send {give.length}, you get {get.length}
          </span>
          <Button tone="volt" onClick={grade}>
            Grade it
          </Button>
        </div>
      )}
    </main>
  );
}

function RosterColumn({
  title,
  side,
  state,
  selected,
  onToggle,
}: {
  title: string;
  side: "you" | "them";
  state: { data?: { players: RosterPlayer[] }; error?: string; loading: boolean };
  selected: string[];
  onToggle: (id: string) => void;
}) {
  const head = side === "you" ? "bg-you text-white" : "bg-them text-ink";
  return (
    <section aria-label={title} className="flex min-w-0 flex-[1_1_400px] flex-col gap-3">
      <div className={`sticker flex items-center justify-between gap-2.5 rounded-2xl px-4 py-3 ${head}`}>
        <h2 className="truncate font-display text-lg font-extrabold md:text-xl">{title}</h2>
        <span className="shrink-0 text-[15px] font-semibold">{selected.length} selected</span>
      </div>
      {state.loading && <Spinner label="Loading roster" />}
      {state.error && <ErrorNote>{state.error}</ErrorNote>}
      <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
        {state.data?.players.map((p) => (
          <PlayerCard key={p.id} p={p} side={side} selected={selected.includes(p.id)} onToggle={() => onToggle(p.id)} />
        ))}
      </div>
    </section>
  );
}

function SlipSide({ label, tone, players, empty }: { label: string; tone: "you" | "them"; players: RosterPlayer[]; empty: string }) {
  const bg = tone === "you" ? "bg-you-tint" : "bg-them-tint border-t-2 border-dashed border-ink";
  const ink = tone === "you" ? "text-you-deep" : "text-them-deep";
  return (
    <div className={`flex flex-col gap-2 px-5 py-3.5 ${bg}`}>
      <span className={`text-sm font-bold ${ink}`}>{label}</span>
      {players.length === 0 && <span className="text-[15px] text-muted">{empty}</span>}
      {players.map((p) => (
        <div key={p.id} className="flex justify-between gap-3 font-semibold">
          <span className="truncate">{p.name}</span>
          <span className="num font-bold">{p.week_pts.toFixed(1)}</span>
        </div>
      ))}
    </div>
  );
}
