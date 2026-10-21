import { ArrowRight, Search } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, ApiError } from "../api";
import { Logo } from "../components/Shell";
import { Button, ErrorNote, Spinner } from "../components/ui";
import { lastLeague } from "../hooks";
import type { League } from "../types";

export function Welcome() {
  const navigate = useNavigate();
  const last = lastLeague();
  const [id, setId] = useState(last?.league ?? "");
  const [league, setLeague] = useState<League | null>(null);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [username, setUsername] = useState("");
  const [found, setFound] = useState<{ league_id: string; name: string; total_rosters: number }[] | null>(null);

  async function load(leagueId: string) {
    setError("");
    setBusy("Loading your league. The very first time takes 2–3 minutes.");
    try {
      setLeague(await api.league(leagueId.trim()));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Something went wrong. Try again.");
    } finally {
      setBusy("");
    }
  }

  async function lookup() {
    setError("");
    setBusy("Looking up your leagues");
    try {
      setFound(await api.leagues(username.trim()));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Something went wrong. Try again.");
    } finally {
      setBusy("");
    }
  }

  return (
    <div className="relative min-h-screen overflow-hidden">
      <div className="pointer-events-none absolute -top-36 -right-32 h-[560px] w-[560px] rounded-full bg-you" aria-hidden="true" />
      <div className="pointer-events-none absolute top-32 right-56 hidden h-60 w-60 rounded-full border-[3px] border-ink bg-them md:block" aria-hidden="true" />
      <div className="pointer-events-none absolute top-[300px] right-32 hidden h-24 w-24 rounded-full border-[3px] border-ink bg-volt md:block" aria-hidden="true" />

      <main className="relative mx-auto flex max-w-[1200px] flex-col gap-10 px-5 py-10 md:px-10 md:py-14">
        <Logo />
        <div className="flex max-w-[720px] flex-col gap-5">
          <h1 className="font-display text-5xl leading-[0.95] font-black tracking-tight md:text-[88px]">Should you make that trade?</h1>
          <p className="max-w-[56ch] text-lg leading-relaxed md:text-xl">
            Paste your Sleeper league and we play out the rest of your season thousands of times, with and without the trade, using your real rosters,
            scoring, schedule and free agents. Then you see how your title odds change, and theirs.
          </p>
        </div>

        <div className="grid max-w-[980px] gap-6 md:grid-cols-[3fr_2fr]">
          <form
            className="sticker-lg flex flex-col gap-4 rounded-[22px] bg-white p-6"
            onSubmit={(e) => {
              e.preventDefault();
              if (id.trim()) load(id);
            }}
          >
            <label htmlFor="league" className="font-display text-xl font-extrabold">
              Your Sleeper league ID
            </label>
            <input
              id="league"
              inputMode="numeric"
              autoComplete="off"
              placeholder="e.g. 1400221038771437568"
              value={id}
              onChange={(e) => setId(e.target.value)}
              className="min-h-14 rounded-2xl border-[2.5px] border-ink px-4 text-lg font-semibold"
            />
            <p className="text-[15px] text-muted">
              Open your league on sleeper.com: the long number in the web address is the ID, like sleeper.com/leagues/<strong>1400221038771437568</strong>.
            </p>
            <Button tone="volt" size="lg" type="submit" disabled={!id.trim() || !!busy}>
              Load my league <ArrowRight size={20} aria-hidden="true" />
            </Button>
          </form>

          <form
            className="sticker-lg flex flex-col gap-4 rounded-[22px] bg-ink p-6 text-white shadow-[6px_6px_0_#FF7A1F]"
            onSubmit={(e) => {
              e.preventDefault();
              if (username.trim()) lookup();
            }}
          >
            <label htmlFor="user" className="font-display text-xl font-extrabold">
              Don’t know the ID?
            </label>
            <input
              id="user"
              autoComplete="username"
              placeholder="Your Sleeper username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="min-h-14 rounded-2xl border-[2.5px] border-white bg-ink px-4 text-lg font-semibold text-white placeholder:text-[#9FA3B5]"
            />
            <Button tone="white" type="submit" disabled={!username.trim() || !!busy}>
              <Search size={18} aria-hidden="true" /> Find my leagues
            </Button>
            {found && (
              <ul className="flex flex-col gap-2">
                {found.length === 0 && <li className="text-[15px]">No leagues for that username this season.</li>}
                {found.map((l) => (
                  <li key={l.league_id}>
                    <button
                      type="button"
                      onClick={() => {
                        setId(l.league_id);
                        load(l.league_id);
                      }}
                      className="flex min-h-12 w-full items-center justify-between rounded-xl border-2 border-white px-4 text-left font-semibold hover:bg-white hover:text-ink"
                    >
                      <span>{l.name.trim() || "Unnamed league"}</span>
                      <span className="text-sm">{l.total_rosters} teams</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </form>
        </div>

        {busy && <Spinner label={busy} />}
        {error && <ErrorNote>{error}</ErrorNote>}

        {league && (
          <section aria-labelledby="pick" className="flex max-w-[980px] flex-col gap-4">
            <h2 id="pick" className="font-display text-3xl font-extrabold">
              Which team is yours?
            </h2>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {league.teams.map((t) => (
                <button
                  key={t.roster_id}
                  type="button"
                  onClick={() => navigate(`/${league.league_id}/${t.roster_id}`)}
                  className="sticker press flex min-h-16 items-center justify-between gap-3 rounded-2xl bg-white px-4 text-left hover:bg-you-tint"
                >
                  <span className="font-bold">{t.name}</span>
                  <span className="num text-sm font-bold text-muted">
                    {t.wins}-{t.losses}
                    {t.ties ? `-${t.ties}` : ""}
                  </span>
                </button>
              ))}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
