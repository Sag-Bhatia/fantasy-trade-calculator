import type {
  FindResult,
  GradeResult,
  League,
  LeagueTrade,
  OddsRow,
  Preview,
  Roster,
  StreamLine,
} from "./types";

export class ApiError extends Error {}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch {
    throw new ApiError("The app's server isn't answering. Is it still running?");
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new ApiError(body?.detail ?? `Something went wrong (${res.status}). Try again.`);
  }
  return res.json() as Promise<T>;
}

const post = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

/** POST a request whose response is NDJSON progress lines followed by one result. */
async function stream<T>(path: string, body: unknown, onProgress: (m: string) => void): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, post(body));
  } catch {
    throw new ApiError("The app's server isn't answering. Is it still running?");
  }
  if (!res.ok || !res.body) {
    const detail = await res.json().catch(() => null);
    throw new ApiError(detail?.detail ?? `Something went wrong (${res.status}). Try again.`);
  }
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += value;
    let nl: number;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line) continue;
      const msg = JSON.parse(line) as StreamLine<T>;
      if (msg.type === "progress") onProgress(msg.message);
      else if (msg.type === "error") throw new ApiError(msg.message);
      else return msg.result;
    }
  }
  throw new ApiError("The grade stopped before finishing. Try again.");
}

export const api = {
  leagues: (username: string) =>
    json<{ league_id: string; name: string; total_rosters: number }[]>(
      `/api/users/${encodeURIComponent(username)}/leagues`,
    ),
  league: (id: string) => json<League>(`/api/league/${encodeURIComponent(id)}`),
  roster: (id: string, team: number) => json<Roster>(`/api/league/${encodeURIComponent(id)}/team/${team}`),
  odds: (id: string) => json<{ sims: number; rows: OddsRow[] }>(`/api/league/${encodeURIComponent(id)}/odds`),
  trades: (id: string) => json<LeagueTrade[]>(`/api/league/${encodeURIComponent(id)}/trades`),
  addFantasypros: (id: string, file: File, week: number) =>
    json<{ matched: number; unmatched: string[] }>(`/api/league/${encodeURIComponent(id)}/fantasypros?week=${week}`, {
      method: "POST",
      headers: { "Content-Type": "text/csv" },
      body: file,
    }),
  removeFantasypros: (id: string) => json<{ ok: boolean }>(`/api/league/${encodeURIComponent(id)}/fantasypros`, { method: "DELETE" }),
  report: () => json<{ markdown: string }>("/api/report"),
  preview: (b: { league_id: string; team: number; partner: number; give: string[]; get: string[] }) =>
    json<Preview>("/api/preview", post(b)),
  grade: (
    b: { league_id: string; team: number; partner: number; give: string[]; get: string[]; sims: number },
    onProgress: (m: string) => void,
  ) => stream<GradeResult>("/api/grade", b, onProgress),
  find: (b: { league_id: string; team: number }, onProgress: (m: string) => void) =>
    stream<FindResult>("/api/find", b, onProgress),
};
