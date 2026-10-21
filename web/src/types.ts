// Shapes returned by the FastAPI layer (docs/api.md).

export type Player = {
  id: string;
  name: string;
  pos: string;
  nfl_team: string | null;
  headshot: string;
};

export type RosterPlayer = Player & {
  slot: string;
  injury: string | null;
  week_pts: number;
  ros_pts: number;
};

export type TeamSummary = {
  roster_id: number;
  name: string;
  wins: number;
  losses: number;
  ties: number;
  points_for: number;
};

export type League = {
  league_id: string;
  name: string;
  season: number;
  week: number;
  num_teams: number;
  scoring: { label: string; per_catch: number; pass_td: number; te_premium: number };
  slots: string[];
  bench: number;
  playoff_teams: number;
  playoff_week_start: number;
  trade_deadline: number;
  deadline_passed: boolean;
  fantasypros: boolean;
  teams: TeamSummary[];
  warnings: string[];
};

export type Roster = { roster_id: number; name: string; record: string; players: RosterPlayer[] };

export type OddsRow = {
  roster_id: number;
  name: string;
  record: string;
  points_for: number;
  ppw: number;
  playoff: number;
  title: number;
};

export type LeagueTrade = {
  week: number;
  sides: { roster_id: number; name: string; got: string[]; value: number }[];
};

export type Delta = { value: number; se: number };

export type TeamResult = {
  roster_id: number;
  name: string;
  title_before: number;
  title_after: number;
  title: Delta;
  playoff_before: number;
  playoff_after: number;
  playoff: Delta;
  ppw: Delta;
  score: number;
  decomposition: { part: string; value: number }[];
  weekly_delta: number[];
  moves: { adds: Player[]; drops: Player[] };
};

export type GradeResult = {
  verdict: "Accept" | "Decline" | "Roughly even";
  sims: number;
  seed: number;
  weeks: number[];
  playoff_week_start: number;
  a: TeamResult;
  b: TeamResult;
  market: { give: number; get: number; gap: number; overpay: boolean; unvalued: string[] } | null;
  acceptance: number | null;
  warnings: string[];
};

export type Preview = { me: number; them: number; shape: string; note: string };

export type Offer = {
  partner: { roster_id: number; name: string; record: string };
  give: Player[];
  get: Player[];
  me: Delta;
  them: Delta;
  me_points: number;
  them_points: number;
  market_gap: number | null;
  moves: { roster_id: number; adds: Player[]; drops: Player[] }[];
};

export type FindResult = { checked: number; rosters: number; offers: Offer[] };

export type StreamLine<T> =
  | { type: "progress"; message: string }
  | { type: "result"; result: T }
  | { type: "error"; message: string };
