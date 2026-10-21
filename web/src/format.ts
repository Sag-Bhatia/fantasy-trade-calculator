// Numbers people read: signed changes, percentages, team colors.

const MINUS = "−";

/** "+3.9", "−1.2", "0.0" */
export function signed(v: number, digits = 1): string {
  const r = Number(v.toFixed(digits));
  if (r === 0) return (0).toFixed(digits);
  return (r > 0 ? "+" : MINUS) + Math.abs(r).toFixed(digits);
}

/** probability 0–1 → "26.4%" */
export const pct = (p: number, digits = 1) => `${(p * 100).toFixed(digits)}%`;

/** probability change 0–1 → "+3.9" (percentage points) */
export const pp = (d: number, digits = 1) => signed(d * 100, digits);

export const initials = (name: string) =>
  name
    .replace(/[^A-Za-z0-9 ]/g, "")
    .split(" ")
    .filter(Boolean)
    .map((w) => w[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();

/** Primary and secondary colors for the jersey stripe on player cards. */
export const TEAM_COLORS: Record<string, [string, string]> = {
  ARI: ["#97233F", "#FFB612"], ATL: ["#A71930", "#12131C"], BAL: ["#241773", "#9E7C0C"], BUF: ["#00338D", "#C60C30"],
  CAR: ["#0085CA", "#101820"], CHI: ["#0B162A", "#C83803"], CIN: ["#FB4F14", "#12131C"], CLE: ["#311D00", "#FF3C00"],
  DAL: ["#003594", "#869397"], DEN: ["#FB4F14", "#002244"], DET: ["#0076B6", "#B0B7BC"], GB: ["#203731", "#FFB612"],
  HOU: ["#03202F", "#A71930"], IND: ["#002C5F", "#A2AAAD"], JAX: ["#006778", "#D7A22A"], KC: ["#E31837", "#FFB81C"],
  LAC: ["#0080C6", "#FFC20E"], LAR: ["#003594", "#FFA300"], LV: ["#000000", "#A5ACAF"], MIA: ["#008E97", "#FC4C02"],
  MIN: ["#4F2683", "#FFC62F"], NE: ["#002244", "#C60C30"], NO: ["#D3BC8D", "#101820"], NYG: ["#0B2265", "#A71930"],
  NYJ: ["#125740", "#FFFFFF"], PHI: ["#004C54", "#A5ACAF"], PIT: ["#FFB612", "#101820"], SEA: ["#69BE28", "#002244"],
  SF: ["#AA0000", "#B3995D"], TB: ["#D50A0A", "#34302B"], TEN: ["#4B92DB", "#0C2340"], WAS: ["#5A1414", "#FFB612"],
};

export const teamColors = (team: string | null): [string, string] =>
  (team && TEAM_COLORS[team]) || ["#A9ADBD", "#12131C"];

/** A steady, distinct color per fantasy team (you are always cobalt). */
const PALETTE = ["#FF7A1F", "#2BD67B", "#B48CFF", "#FFC83D", "#FF5C7A", "#29C5F6", "#9BE15D", "#F59BD5", "#FFA45C", "#7BD3C9", "#C9B6FF", "#FFE066", "#8FB3FF", "#FF9E80", "#A3E4A3"];
export const managerColor = (rosterId: number, me: number) => (rosterId === me ? "#3557FF" : PALETTE[(rosterId * 7) % PALETTE.length]);
