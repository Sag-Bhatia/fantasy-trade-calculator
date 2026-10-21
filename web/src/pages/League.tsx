import { useState } from "react";
import { Link } from "react-router-dom";
import { api, ApiError } from "../api";
import { useShell } from "../components/Shell";
import { Button, Card, ErrorNote, Segmented, Spinner } from "../components/ui";
import { initials, managerColor, pct } from "../format";
import { useLoad } from "../hooks";

export function LeaguePage() {
  const { league, team, base } = useShell();
  const odds = useLoad(`odds:${league.league_id}`, () => api.odds(league.league_id));
  const trades = useLoad(`trades:${league.league_id}`, () => api.trades(league.league_id));
  const [metric, setMetric] = useState<"playoff" | "title">("playoff");
  const [pick, setPick] = useState<number | null>(null);

  const rows = [...(odds.data?.rows ?? [])].sort((a, b) => (metric === "playoff" ? b.playoff - a.playoff : b.title - a.title));
  const max = metric === "playoff" ? 1 : Math.max(0.3, ...rows.map((r) => r.title));
  const sel = rows.find((r) => r.roster_id === (pick ?? team)) ?? rows[0];
  const sc = league.scoring;

  return (
    <main className="mx-auto flex max-w-[1400px] flex-col gap-6 px-5 py-7 md:px-10">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="flex flex-col gap-2">
          <h1 className="font-display text-4xl leading-[0.95] font-black tracking-tight md:text-[56px]">The playoff race</h1>
          <p className="text-[17px] text-muted">
            {league.num_teams} teams, {sc.label.toLowerCase()}, top {league.playoff_teams} make the playoffs in week {league.playoff_week_start}.
            {league.trade_deadline ? ` Trade deadline: week ${league.trade_deadline}.` : ""}
          </p>
        </div>
        <Segmented
          label="Show"
          value={metric}
          onChange={setMetric}
          options={[
            { value: "playoff", label: "Playoff odds" },
            { value: "title", label: "Title odds" },
          ]}
        />
      </div>

      <div className="flex flex-wrap items-start gap-6">
        <Card className="flex min-w-0 flex-[3_1_680px] flex-col gap-1.5 p-3.5" aria-label="Standings">
          {odds.loading && <Spinner label="Simulating the rest of the season" />}
          {odds.error && <ErrorNote>{odds.error}</ErrorNote>}
          {rows.map((r, i) => {
            const me = r.roster_id === team;
            const on = sel?.roster_id === r.roster_id;
            const v = metric === "playoff" ? r.playoff : r.title;
            const color = managerColor(r.roster_id, team);
            return (
              <div key={r.roster_id}>
                {metric === "playoff" && i === league.playoff_teams && (
                  <div className="my-1.5 flex items-center gap-2.5">
                    <div className="flex-1 border-t-[3px] border-dashed border-ink" />
                    <span className="num rounded-full border-2 border-ink bg-volt px-3 py-1 text-[13px] font-extrabold">Playoff line</span>
                    <div className="flex-1 border-t-[3px] border-dashed border-ink" />
                  </div>
                )}
                <button
                  type="button"
                  aria-pressed={on}
                  onClick={() => setPick(r.roster_id)}
                  className={`grid min-h-[54px] w-full grid-cols-[24px_40px_minmax(0,1fr)_60px] items-center gap-x-3 gap-y-1 rounded-2xl border-[2.5px] px-3 py-1.5 text-left md:grid-cols-[34px_44px_minmax(140px,1.2fr)_64px_minmax(0,2fr)_72px] ${
                    on ? "border-ink shadow-[4px_4px_0_#12131C]" : "border-transparent"
                  } ${me ? "bg-you-tint" : on ? "bg-court" : "bg-white"}`}
                >
                  <span className="num text-[15px] font-extrabold text-muted">{i + 1}</span>
                  <span
                    className="flex h-10 w-10 items-center justify-center rounded-full border-2 border-ink font-display text-xs font-extrabold"
                    style={{ background: color, color: me ? "#fff" : "#12131C" }}
                    aria-hidden="true"
                  >
                    {initials(r.name)}
                  </span>
                  <span className="flex min-w-0 flex-col">
                    <span className="truncate font-bold">
                      {r.name}
                      {me ? " (you)" : ""}
                    </span>
                    <span className="text-[13px] text-muted">{r.ppw.toFixed(1)} pts a week</span>
                  </span>
                  <span className="num hidden text-[15px] font-bold md:block">{r.record}</span>
                  <span className="relative col-span-2 col-start-3 block h-[18px] overflow-hidden rounded-full border-2 border-ink bg-line max-md:order-last md:col-span-1 md:col-start-auto">
                    <span
                      className="absolute inset-y-0 left-0 block border-r-2 border-ink"
                      style={{ width: `${Math.min((v / max) * 100, 100)}%`, background: me ? "#3557FF" : metric === "playoff" && i >= league.playoff_teams ? "#C5C9D8" : color }}
                    />
                  </span>
                  <span className="num text-right font-extrabold">{pct(v)}</span>
                </button>
              </div>
            );
          })}
          {odds.data && (
            <p className="mx-3 mt-2.5 text-sm text-muted">
              From {odds.data.sims.toLocaleString()} simulated seasons with today’s rosters and each team’s real remaining schedule. Tap a team for details.
            </p>
          )}
        </Card>

        <aside className="flex min-w-0 flex-[1_1_320px] flex-col gap-5">
          {sel && (
            <section
              aria-labelledby="pick"
              className="sticker-lg flex flex-col gap-2.5 rounded-[24px] p-5"
              style={{ background: managerColor(sel.roster_id, team), color: sel.roster_id === team ? "#fff" : "#12131C" }}
            >
              <h2 id="pick" className="font-display text-xl font-extrabold">
                {sel.roster_id === team ? `You, ${sel.name}` : `${sel.name} (${sel.record})`}
              </h2>
              <div className="flex gap-6">
                <div className="flex flex-col">
                  <span className="text-[13px] font-semibold">Playoffs</span>
                  <span className="num text-3xl font-black">{pct(sel.playoff)}</span>
                </div>
                <div className="flex flex-col">
                  <span className="text-[13px] font-semibold">Title</span>
                  <span className="num text-3xl font-black">{pct(sel.title)}</span>
                </div>
              </div>
              <Link
                to={sel.roster_id === team ? `${base}/find` : `${base}?partner=${sel.roster_id}`}
                className="sticker press mt-1 inline-flex min-h-12 items-center self-start rounded-[14px] bg-volt px-4 font-bold text-ink"
              >
                {sel.roster_id === team ? "Find trades for me" : `Build a trade with ${sel.name}`}
              </Link>
            </section>
          )}
          <Card className="flex flex-col gap-3.5 p-5" aria-labelledby="tr">
            <h2 id="tr" className="font-display text-xl font-extrabold">
              Trades this season
            </h2>
            {trades.loading && <Spinner label="Loading trades" />}
            {trades.data?.length === 0 && <p className="text-[15px]">No trades in this league yet.</p>}
            {trades.data?.slice(0, 8).map((t, i) => (
              <div key={i} className="flex flex-col gap-1.5 border-t-2 border-dashed border-ink pt-2.5">
                <span className="num text-[13px] font-bold text-muted">Week {t.week}</span>
                {t.sides.map((s) => (
                  <div key={s.roster_id} className="flex justify-between gap-2.5 text-[15px] leading-snug">
                    <span>
                      <strong>{s.name}</strong> got {s.got.join(", ") || "draft picks"}
                    </span>
                    <span className="num shrink-0 font-bold">{s.value.toLocaleString()}</span>
                  </div>
                ))}
              </div>
            ))}
            <span className="text-[13px] text-muted">Today’s FantasyCalc market value of what each team received.</span>
          </Card>
          {league.fantasypros_upload && <FantasyPros leagueId={league.league_id} on={league.fantasypros} week={Math.min(league.week, 18)} />}
        </aside>
      </div>
    </main>
  );
}

/** Optional: blend a FantasyPros projections CSV the user exported (they have no free feed). */
function FantasyPros({ leagueId, on, week: thisWeek }: { leagueId: string; on: boolean; week: number }) {
  const [file, setFile] = useState<File | null>(null);
  const [week, setWeek] = useState(thisWeek);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [done, setDone] = useState<{ matched: number; unmatched: string[] } | null>(null);
  // every number changes with the blend, so start fresh rather than patch cached results
  const reload = () => window.location.reload();

  async function run(work: () => Promise<unknown>) {
    setBusy(true);
    setError("");
    try {
      await work();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Something went wrong. Try again.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card className="flex flex-col gap-3 p-5" aria-labelledby="fp">
      <h2 id="fp" className="font-display text-xl font-extrabold">
        FantasyPros projections
      </h2>
      {on ? (
        <>
          <p className="text-[15px]">Your FantasyPros file is part of the projection blend until you restart the app.</p>
          <Button tone="white" disabled={busy} onClick={() => run(() => api.removeFantasypros(leagueId).then(reload))}>
            Stop using the file
          </Button>
        </>
      ) : done ? (
        <>
          <p className="text-[15px]">
            {done.matched.toLocaleString()} projection rows matched.
            {done.unmatched.length > 0 && ` ${done.unmatched.length} didn’t match a player and were left out: ${done.unmatched.slice(0, 8).join(", ")}${done.unmatched.length > 8 ? ", and more" : ""}.`}
          </p>
          <Button tone="volt" onClick={reload}>
            Reload the numbers with it
          </Button>
        </>
      ) : (
        <>
          <p className="text-[15px]">Optional. Export a weekly projections CSV from fantasypros.com and add it here to mix their consensus into the blend.</p>
          <label className="flex flex-col gap-1 text-sm font-bold">
            CSV file
            <input type="file" accept=".csv,text/csv" onChange={(e) => setFile(e.target.files?.[0] ?? null)} className="text-[15px] font-normal" />
          </label>
          <label className="flex items-center gap-2.5 text-sm font-bold">
            Week it projects
            <input
              type="number"
              min={1}
              max={18}
              value={week}
              onChange={(e) => setWeek(Number(e.target.value))}
              className="min-h-11 w-20 rounded-xl border-[2.5px] border-ink bg-white px-2.5 text-[15px]"
            />
          </label>
          <Button tone="volt" disabled={!file || busy || week < 1 || week > 18} onClick={() => file && run(() => api.addFantasypros(leagueId, file, week).then(setDone))}>
            {busy ? "Adding it…" : "Use this file"}
          </Button>
        </>
      )}
      {error && <ErrorNote>{error}</ErrorNote>}
    </Card>
  );
}
