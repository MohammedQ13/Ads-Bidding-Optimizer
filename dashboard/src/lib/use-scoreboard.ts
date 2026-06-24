"use client";

import { useEffect, useRef, useState } from "react";
import type { Snapshot } from "./types";

export type ConnState = "connecting" | "live" | "offline";

// one point on the rolling time series we keep for the live charts
export interface HistPoint {
  t: number; // tick index
  aps: number;
  clearing: number;
  fleetP99: number; // worst per-company round-trip p99, fleet-wide
  inferenceP50: number; // fleet-average ONNX inference p50 (microseconds)
  fallbackRate: number;
  fleetServed: number; // total real req/s the fleet handles (sum of served_per_sec)
  [companyId: string]: number; // cumulative profit per company
}

const MAX_POINTS = 120; // ~2 min of history at 1 point/sec
const POLL_MS = 1000;

// Polls the same-origin /api/stats proxy once a second, keeps the latest
// snapshot, and accumulates a rolling history (profit per company, throughput,
// clearing price, fleet p99 latency, fallback rate) for the time-series charts.
export function useScoreboard(mode: "live" | "maxcap" = "live") {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [history, setHistory] = useState<HistPoint[]>([]);
  const [state, setState] = useState<ConnState>("connecting");
  const tick = useRef(0);
  const lastAuctions = useRef(0);

  useEffect(() => {
    let stopped = false;
    // switching mode starts a fresh history so the charts don't splice two runs
    tick.current = 0;
    lastAuctions.current = 0;
    setHistory([]);

    const apply = (snap: Snapshot) => {
      setSnapshot(snap);
      // a looping replay restarts at frame 0, where cumulative counters drop back
      // down; start a fresh history so the charts don't show a downward step
      if (snap.auctions < lastAuctions.current) {
        tick.current = 0;
        setHistory([]);
      }
      lastAuctions.current = snap.auctions;
      let fleetP99 = 0;
      for (const id of Object.keys(snap.companies)) {
        const p = snap.companies[id].rtt_us?.p99 ?? 0;
        if (p > fleetP99) fleetP99 = p;
      }
      let avgFallback = 0;
      const ids = Object.keys(snap.companies);
      if (ids.length > 0) {
        let s = 0;
        for (const id of ids) s += snap.companies[id].fallback_rate ?? 0;
        avgFallback = s / ids.length;
      }
      // fleet-average inference p50, only over servers reporting a real number
      let infSum = 0;
      let infN = 0;
      let fleetServed = 0;
      for (const id of ids) {
        const p = snap.companies[id].inference_us?.p50 ?? 0;
        if (p > 0) {
          infSum += p;
          infN++;
        }
        fleetServed += snap.companies[id].served_per_sec ?? 0;
      }
      const point: HistPoint = {
        t: tick.current++,
        aps: snap.auctions_per_sec,
        clearing: snap.last_clearing_price,
        fleetP99,
        inferenceP50: infN > 0 ? infSum / infN : 0,
        fallbackRate: avgFallback,
        fleetServed,
      };
      for (const [id, c] of Object.entries(snap.companies)) {
        point[id] = c.profit;
      }
      setHistory((prev) => {
        const next = [...prev, point];
        return next.length > MAX_POINTS ? next.slice(-MAX_POINTS) : next;
      });
    };

    const url = mode === "maxcap" ? "/api/stats?mode=maxcap" : "/api/stats";
    const poll = async () => {
      try {
        const r = await fetch(url, { cache: "no-store" });
        if (!r.ok) throw new Error("backend");
        const snap = (await r.json()) as Snapshot;
        if (stopped) return;
        if (!snap.companies) throw new Error("empty");
        apply(snap);
        setState("live");
      } catch {
        if (!stopped) setState("offline");
      }
    };

    poll();
    const id = setInterval(poll, POLL_MS);
    return () => {
      stopped = true;
      clearInterval(id);
    };
  }, [mode]);

  return { snapshot, history, state };
}
