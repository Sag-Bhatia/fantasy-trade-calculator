import { animate, useReducedMotion } from "motion/react";
import { useEffect, useRef, useState } from "react";
import { ApiError } from "./api";

const cache = new Map<string, unknown>();

export type Async<T> = { data: T | undefined; error: string | undefined; loading: boolean };

/** Load once per key and remember the answer for the rest of the session. */
export function useLoad<T>(key: string | null, load: () => Promise<T>): Async<T> {
  const [state, setState] = useState<Async<T>>(() => ({
    data: key ? (cache.get(key) as T | undefined) : undefined,
    error: undefined,
    loading: !!key && !cache.has(key),
  }));
  useEffect(() => {
    if (!key) return;
    if (cache.has(key)) {
      setState({ data: cache.get(key) as T, error: undefined, loading: false });
      return;
    }
    let live = true;
    setState({ data: undefined, error: undefined, loading: true });
    load()
      .then((d) => {
        cache.set(key, d);
        if (live) setState({ data: d, error: undefined, loading: false });
      })
      .catch((e: unknown) => {
        const msg = e instanceof ApiError ? e.message : "Something went wrong. Try again.";
        if (live) setState({ data: undefined, error: msg, loading: false });
      });
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return state;
}

export const remember = (key: string, value: unknown) => cache.set(key, value);
export const recall = <T,>(key: string) => cache.get(key) as T | undefined;

/** Animate 0 → 1 once when `run` changes; instant for people who prefer less motion. */
export function useCountUp(run: unknown, ms = 1400): number {
  const reduce = useReducedMotion();
  const [t, setT] = useState(reduce ? 1 : 0);
  const first = useRef(true);
  useEffect(() => {
    if (reduce) {
      setT(1);
      return;
    }
    first.current = false;
    setT(0);
    const c = animate(0, 1, { duration: ms / 1000, ease: [0.16, 1, 0.3, 1], onUpdate: setT });
    return () => c.stop();
  }, [run, ms, reduce]);
  return t;
}

const LAST = "ffcalc.last";
export function lastLeague(): { league: string; team: number } | null {
  try {
    const v = JSON.parse(localStorage.getItem(LAST) ?? "null");
    return v && typeof v.league === "string" && typeof v.team === "number" ? v : null;
  } catch {
    return null;
  }
}
export function saveLastLeague(league: string, team: number) {
  try {
    localStorage.setItem(LAST, JSON.stringify({ league, team }));
  } catch {
    /* private mode: nothing to remember */
  }
}
