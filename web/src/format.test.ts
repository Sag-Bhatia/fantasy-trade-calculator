import { describe, expect, it } from "vitest";
import { initials, pct, pp, signed, teamColors } from "./format";

describe("number formatting", () => {
  it("signs changes with a real minus and no -0.0", () => {
    expect(signed(3.94)).toBe("+3.9");
    expect(signed(-1.25)).toBe("−1.3");
    expect(signed(-0.04)).toBe("0.0");
    expect(signed(48.2, 0)).toBe("+48");
  });
  it("shows probabilities and percentage-point changes", () => {
    expect(pct(0.264)).toBe("26.4%");
    expect(pp(0.039)).toBe("+3.9");
  });
  it("makes initials and falls back to grey for unknown teams", () => {
    expect(initials("Ja'Marr Chase")).toBe("JC");
    expect(initials("Kansas City Chiefs")).toBe("KC");
    expect(teamColors(null)[0]).toBe("#A9ADBD");
    expect(teamColors("LAR")).toEqual(["#003594", "#FFA300"]);
  });
});
