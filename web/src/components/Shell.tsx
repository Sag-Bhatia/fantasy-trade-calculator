import { ArrowLeftRight } from "lucide-react";
import { Navigate, NavLink, Outlet, ScrollRestoration, useNavigate, useOutletContext, useParams } from "react-router-dom";
import { api } from "../api";
import { saveLastLeague, useLoad } from "../hooks";
import type { League } from "../types";
import { ErrorNote, Spinner } from "./ui";

export type ShellCtx = { league: League; team: number; base: string };
export const useShell = () => useOutletContext<ShellCtx>();

export function Logo() {
  return (
    <span className="flex items-center gap-2.5">
      <span className="flex h-10 w-10 items-center justify-center rounded-[11px] border-[2.5px] border-ink bg-volt shadow-[2px_2px_0_#12131C]">
        <ArrowLeftRight size={20} strokeWidth={2.6} aria-hidden="true" />
      </span>
      <span className="font-display text-[19px] font-extrabold">Trade Calculator</span>
    </span>
  );
}

const TABS = [
  { to: "", label: "Grade a trade", end: true },
  { to: "find", label: "Find trades" },
  { to: "league", label: "Your league" },
  { to: "accuracy", label: "How accurate is it?" },
];

export function Shell() {
  const { league: leagueId = "", team: teamParam = "" } = useParams();
  const navigate = useNavigate();
  const team = Number(teamParam);
  const league = useLoad(`league:${leagueId}`, () => api.league(leagueId));
  const base = `/${leagueId}/${team}`;

  if (league.loading) {
    return (
      <div className="flex min-h-screen items-center justify-center p-8">
        <Spinner label="Loading your league. The very first time takes 2–3 minutes while it learns from past seasons." />
      </div>
    );
  }
  if (league.error || !league.data) {
    return (
      <div className="mx-auto flex min-h-screen max-w-xl flex-col justify-center gap-5 p-8">
        <Logo />
        <ErrorNote>{league.error}</ErrorNote>
        <a className="font-bold" href="/">
          Try a different league
        </a>
      </div>
    );
  }
  const lg = league.data;
  const me = lg.teams.find((t) => t.roster_id === team) ?? lg.teams[0];
  if (me.roster_id !== team) return <Navigate to={`/${leagueId}/${me.roster_id}`} replace />;
  saveLastLeague(leagueId, me.roster_id);

  return (
    <div className="min-h-screen pb-24 md:pb-0">
      <nav aria-label="Main" className="border-b-[2.5px] border-ink bg-white">
        <div className="mx-auto flex max-w-[1400px] flex-wrap items-center gap-x-7 gap-y-3 px-5 py-3.5 md:px-10">
          <NavLink to={base} aria-label="Trade Calculator home">
            <Logo />
          </NavLink>
          <div className="hidden flex-wrap gap-1.5 md:flex">
            {TABS.map((t) => (
              <NavLink
                key={t.to}
                to={t.to ? `${base}/${t.to}` : base}
                end={t.end}
                className={({ isActive }) =>
                  `rounded-full px-4 py-2.5 text-[15px] font-semibold ${isActive ? "bg-ink text-white" : "hover:bg-court"}`
                }
              >
                {t.label}
              </NavLink>
            ))}
          </div>
          <label className="ml-auto flex items-center gap-2.5 text-[15px] font-semibold">
            <span className="h-7 w-7 rounded-full border-2 border-ink bg-you" aria-hidden="true" />
            <span className="sr-only">Your team</span>
            <select
              value={me.roster_id}
              onChange={(e) => navigate(`/${leagueId}/${e.target.value}`)}
              className="min-h-11 rounded-xl border-2 border-ink bg-white px-3 font-semibold"
            >
              {lg.teams.map((t) => (
                <option key={t.roster_id} value={t.roster_id}>
                  {t.name} ({t.wins}-{t.losses}
                  {t.ties ? `-${t.ties}` : ""})
                </option>
              ))}
            </select>
          </label>
        </div>
      </nav>
      {lg.deadline_passed && (
        <div className="bg-ink px-5 py-2.5 text-center text-[15px] font-semibold text-volt">
          This league’s trade deadline (week {lg.trade_deadline}) has passed, so trades here are hypothetical.
        </div>
      )}
      <Outlet context={{ league: lg, team: me.roster_id, base } satisfies ShellCtx} />
      <ScrollRestoration />
      <nav aria-label="Main" className="fixed inset-x-0 bottom-0 z-20 grid grid-cols-4 border-t-[2.5px] border-ink bg-white px-1 pt-1.5 pb-5 md:hidden">
        {TABS.map((t) => (
          <NavLink
            key={t.to}
            to={t.to ? `${base}/${t.to}` : base}
            end={t.end}
            className={({ isActive }) =>
              `flex min-h-[52px] flex-col items-center justify-center rounded-xl text-[12px] font-bold ${isActive ? "bg-volt" : "text-muted"}`
            }
          >
            {t.label.replace("How accurate is it?", "Accuracy").replace("Grade a trade", "Grade").replace("Find trades", "Find").replace("Your league", "League")}
          </NavLink>
        ))}
      </nav>
    </div>
  );
}
