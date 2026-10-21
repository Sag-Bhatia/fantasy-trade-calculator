import { Check } from "lucide-react";
import type { RosterPlayer } from "../types";
import { Avatar, Jersey } from "./ui";

const TAG: Record<string, string> = { Questionable: "Q", Doubtful: "D", Out: "Out", IR: "IR", PUP: "PUP", Sus: "Sus" };

export function PlayerCard({
  p,
  side,
  selected,
  onToggle,
}: {
  p: RosterPlayer;
  side: "you" | "them";
  selected: boolean;
  onToggle: () => void;
}) {
  const ring = side === "you" ? "#3557FF" : "#FF7A1F";
  const tint = side === "you" ? "bg-you-tint" : "bg-them-tint";
  const tag = p.injury ? (TAG[p.injury] ?? p.injury) : null;
  return (
    <button
      type="button"
      aria-pressed={selected}
      onClick={onToggle}
      className={`group flex min-h-16 items-stretch overflow-hidden rounded-[14px] border-[2.5px] border-ink text-left transition-[transform,box-shadow,background-color] duration-150 motion-reduce:transition-none ${
        selected ? `${tint} -translate-x-0.5 -translate-y-0.5` : "bg-white hover:-translate-y-0.5"
      }`}
      style={{ boxShadow: selected ? `5px 5px 0 ${ring}` : "3px 3px 0 #12131C" }}
    >
      <Jersey team={p.nfl_team} className="w-3 shrink-0" />
      <span className="flex min-w-0 flex-1 items-center gap-2 px-2 py-2">
        <span className="relative">
          <Avatar player={p} size={36} />
          {selected && (
            <span className="absolute -right-1 -bottom-1 flex h-5 w-5 items-center justify-center rounded-full border-2 border-ink" style={{ background: ring }}>
              <Check size={12} strokeWidth={3.5} color={side === "you" ? "#fff" : "#12131C"} aria-hidden="true" />
            </span>
          )}
        </span>
        <span className="flex min-w-0 flex-1 flex-col">
          <span className="text-[15px] leading-tight font-bold break-words">{p.name}</span>
          <span className="text-[13px] whitespace-nowrap text-muted">
            {p.slot === "BN" ? "Bench" : p.slot === "IR" ? "IR" : p.slot.replace("_", " ")}, {p.nfl_team ?? "free agent"}
          </span>
        </span>
        <span className="flex flex-col items-end gap-0.5">
          <span className="num text-[17px] font-extrabold">{p.week_pts.toFixed(1)}</span>
          {tag && <span className="rounded-[5px] bg-ink px-1.5 text-[11px] font-bold text-volt">{tag}</span>}
        </span>
      </span>
      <span className="sr-only">{selected ? ", in the trade" : ""}</span>
    </button>
  );
}
