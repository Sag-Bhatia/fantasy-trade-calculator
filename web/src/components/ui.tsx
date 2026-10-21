import { useState, type ButtonHTMLAttributes, type ReactNode } from "react";
import type { Player } from "../types";
import { initials, teamColors } from "../format";

type Tone = "volt" | "white" | "ink" | "you" | "them";
const TONES: Record<Tone, string> = {
  volt: "bg-volt text-ink",
  white: "bg-white text-ink",
  ink: "bg-ink text-white",
  you: "bg-you text-white",
  them: "bg-them text-ink",
};

export function Button({
  tone = "white",
  size = "md",
  className = "",
  children,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { tone?: Tone; size?: "md" | "lg"; children: ReactNode }) {
  const sz = size === "lg" ? "min-h-[60px] px-7 text-lg font-display font-extrabold rounded-2xl" : "min-h-12 px-5 font-bold rounded-[14px]";
  return (
    <button
      {...rest}
      className={`sticker press inline-flex items-center justify-center gap-2 ${sz} ${TONES[tone]} disabled:opacity-50 disabled:active:translate-0 ${className}`}
    >
      {children}
    </button>
  );
}

/** Headshot from Sleeper's CDN, initials if it can't load. */
export function Avatar({ player, size = 40 }: { player: Pick<Player, "name" | "headshot" | "nfl_team">; size?: number }) {
  const [broken, setBroken] = useState(false);
  const [c1] = teamColors(player.nfl_team);
  return (
    <span
      className="relative inline-flex shrink-0 items-center justify-center overflow-hidden rounded-full border-2 border-ink bg-white font-display text-[11px] font-extrabold"
      style={{ width: size, height: size, boxShadow: `inset 0 -${size / 3}px 0 ${c1}22` }}
      aria-hidden="true"
    >
      {broken ? (
        initials(player.name)
      ) : (
        <img src={player.headshot} alt="" className="h-full w-full object-cover" loading="lazy" onError={() => setBroken(true)} />
      )}
    </span>
  );
}

export function Jersey({ team, className = "" }: { team: string | null; className?: string }) {
  const [c1, c2] = teamColors(team);
  return <span className={`jersey block ${className}`} style={{ "--c1": c1, "--c2": c2 } as React.CSSProperties} aria-hidden="true" />;
}

export function Card({ className = "", children, as: Tag = "section", ...rest }: { className?: string; children: ReactNode; as?: "section" | "article" | "div" | "aside" } & React.HTMLAttributes<HTMLElement>) {
  return (
    <Tag {...rest} className={`sticker-lg rounded-[22px] bg-white ${className}`}>
      {children}
    </Tag>
  );
}

export function Pill({ tone, children }: { tone: "up" | "down" | "plain"; children: ReactNode }) {
  const cls = tone === "up" ? "bg-up-tint text-[#065C34]" : tone === "down" ? "bg-down-tint text-[#A3122F]" : "bg-white";
  return <span className={`num inline-flex items-center rounded-full border-2 border-ink px-3 py-1 text-sm font-extrabold ${cls}`}>{children}</span>;
}

export function Segmented<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: { value: T; label: ReactNode; activeClass?: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div role="group" aria-label={label} className="inline-flex gap-1.5 rounded-2xl border-[2.5px] border-ink bg-white p-1.5">
      {options.map((o) => {
        const on = o.value === value;
        return (
          <button
            key={o.value}
            type="button"
            aria-pressed={on}
            onClick={() => onChange(o.value)}
            className={`min-h-10 rounded-[11px] px-4 text-[15px] font-bold transition-colors ${on ? (o.activeClass ?? "bg-ink text-volt") : "text-ink hover:bg-court"}`}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

export function Spinner({ label }: { label: string }) {
  return (
    <div role="status" className="flex items-center gap-3 font-semibold">
      <span className="h-5 w-5 animate-spin rounded-full border-[3px] border-ink border-t-volt motion-reduce:animate-none" aria-hidden="true" />
      {label}
    </div>
  );
}

export function ErrorNote({ children }: { children: ReactNode }) {
  return (
    <div role="alert" className="sticker rounded-2xl bg-down-tint px-5 py-4 font-semibold text-[#7A0E24]">
      {children}
    </div>
  );
}
