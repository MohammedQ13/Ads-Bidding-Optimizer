"use client";

import { Gauge } from "lucide-react";
import { Panel, Pill } from "./ui/panel";
import { metaFor, type Snapshot } from "@/lib/types";
import { compact, us } from "@/lib/utils";

// RED-style latency (rate / fallback / duration per service) plus the inference
// distribution: a histogram of recent inference times with p50/p95/p99 marks,
// then one row per bidder against the bid deadline. Note the middle column is the
// fallback rate (model bypassed for the heuristic), not an RPC error rate.
export function LatencyPanel({ snapshot }: { snapshot: Snapshot | null }) {
  const ids = snapshot?.order?.length
    ? snapshot.order
    : snapshot
    ? Object.keys(snapshot.companies).sort()
    : [];
  const deadlineUs = (snapshot?.deadline_ms ?? 10) * 1000;

  // collect inference-time samples from the recent auctions for the distribution
  const samples: number[] = [];
  for (const ev of snapshot?.events ?? []) {
    for (const b of ev.bids) {
      if (!b.paced && b.inf_us > 0) samples.push(b.inf_us);
    }
  }

  return (
    <Panel
      icon={<Gauge className="h-4 w-4" />}
      title="Latency · RED"
      subtitle={`rate, fallback, duration vs the ${snapshot?.deadline_ms ?? 10}ms deadline`}
      info="RED method per server: Rate (bids handled), Errors (here the fallback rate — model bypassed for the heuristic, not RPC errors), Duration (round-trip p50/p95/p99 drawn against the 10ms bid deadline). The top chart is the ONNX inference-time distribution."
      right={<Pill tone="info">p50 · p95 · p99</Pill>}
    >
      {ids.length === 0 ? (
        <div className="h-40 grid place-items-center text-[11px] text-muted-foreground">
          waiting for latency samples…
        </div>
      ) : (
        <div className="space-y-4">
          <Distribution samples={samples} />

          {/* RED rows, front-of-path first */}
          <div>
            <div className="grid grid-cols-[120px_64px_64px_1fr] gap-2 px-1 pb-1 text-[9px] uppercase tracking-[0.1em] text-label border-b border-hairline">
              <span>service</span>
              <span className="text-right">rate</span>
              <span className="text-right">fallback</span>
              <span>duration (round-trip vs deadline)</span>
            </div>
            {ids.map((id) => {
              const c = snapshot!.companies[id];
              const m = metaFor(id);
              const fbPct = (c.fallback_rate ?? 0) * 100;
              return (
                <div
                  key={id}
                  className="grid grid-cols-[120px_64px_64px_1fr] gap-2 px-1 py-1.5 items-center border-b border-hairline/60"
                >
                  <span className="text-[11px] font-medium flex items-center gap-1.5 mono truncate">
                    <span className="h-2 w-2 rounded-sm shrink-0" style={{ background: m.color }} />
                    {m.short.toLowerCase()}
                  </span>
                  <span className="text-right text-[10.5px] tnum mono text-muted-foreground">
                    {compact(c.bids)}
                  </span>
                  <span
                    className={`text-right text-[10.5px] tnum mono ${
                      fbPct > 1 ? "text-warning" : "text-muted-foreground"
                    }`}
                  >
                    {fbPct.toFixed(1)}%
                  </span>
                  <DurationTrack color={m.color} deadlineUs={deadlineUs} lat={c.rtt_us} />
                </div>
              );
            })}
          </div>
        </div>
      )}
    </Panel>
  );
}

// Histogram of inference times across the recent feed, with p50/p95/p99 marks.
function Distribution({ samples }: { samples: number[] }) {
  if (samples.length < 4) {
    return (
      <div className="h-20 grid place-items-center text-[10px] text-muted-foreground border border-hairline rounded-md">
        collecting inference samples…
      </div>
    );
  }
  const sorted = [...samples].sort((a, b) => a - b);
  const q = (p: number) => sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
  const lo = sorted[0];
  const hi = sorted[sorted.length - 1];
  const p50 = q(0.5);
  const p95 = q(0.95);
  const p99 = q(0.99);

  // linear buckets across the observed range
  const N = 28;
  const span = Math.max(hi - lo, 1);
  const counts = new Array(N).fill(0);
  for (const v of samples) {
    let idx = Math.floor(((v - lo) / span) * N);
    if (idx < 0) idx = 0;
    if (idx >= N) idx = N - 1;
    counts[idx]++;
  }
  const maxCount = Math.max(1, ...counts);
  const xOf = (v: number) => ((v - lo) / span) * 100;

  return (
    <div>
      <div className="flex items-center justify-between mb-1.5 text-[9px] uppercase tracking-[0.1em] text-label">
        <span>ONNX inference distribution · {compact(samples.length)} samples</span>
        <span className="mono normal-case text-muted-foreground">
          p50 {us(p50)} · p95 {us(p95)} · p99 {us(p99)}
        </span>
      </div>
      <div className="relative h-20">
        {/* bars */}
        <div className="absolute inset-0 flex items-end gap-px">
          {counts.map((n, i) => (
            <div
              key={i}
              className="flex-1 rounded-t-[1px] bg-primary/45"
              style={{ height: `${(n / maxCount) * 100}%` }}
            />
          ))}
        </div>
        {/* percentile markers */}
        <Marker xPct={xOf(p50)} label="p50" tone="muted" />
        <Marker xPct={xOf(p95)} label="p95" tone="muted" />
        <Marker xPct={xOf(p99)} label="p99" tone="warn" />
      </div>
      <div className="flex justify-between text-[8.5px] mono text-label mt-1">
        <span>{us(lo)}</span>
        <span>{us(hi)}</span>
      </div>
    </div>
  );
}

function Marker({ xPct, label, tone }: { xPct: number; label: string; tone: "muted" | "warn" }) {
  const color = tone === "warn" ? "var(--warning)" : "var(--muted-foreground)";
  return (
    <div className="absolute top-0 bottom-0" style={{ left: `${xPct}%` }}>
      <div className="h-full border-l border-dashed" style={{ borderColor: color, opacity: 0.7 }} />
      <span
        className="absolute top-0 text-[8px] mono pl-0.5 whitespace-nowrap"
        style={{ color }}
      >
        {label}
      </span>
    </div>
  );
}

// A round-trip percentile bar drawn against the deadline so headroom is obvious:
// nested p99 / p95 / p50 fills, then the p50/p95/p99 values.
function DurationTrack({
  color,
  deadlineUs,
  lat,
}: {
  color: string;
  deadlineUs: number;
  lat: { p50: number; p95: number; p99: number; avg: number };
}) {
  const w = (v: number) => `${Math.min(100, Math.max(0.5, (v / deadlineUs) * 100))}%`;
  return (
    <div className="flex items-center gap-2">
      <div className="relative flex-1 h-3.5 rounded-sm bg-muted overflow-hidden">
        <div className="absolute inset-y-0 left-0 rounded-sm" style={{ width: w(lat.p99), background: `${color}33` }} />
        <div className="absolute inset-y-0 left-0 rounded-sm" style={{ width: w(lat.p95), background: `${color}66` }} />
        <div className="absolute inset-y-0 left-0 rounded-sm" style={{ width: w(lat.p50), background: color }} />
      </div>
      <span className="w-[120px] shrink-0 text-right text-[9.5px] tnum mono text-muted-foreground">
        {us(lat.p50)} / {us(lat.p95)} / {us(lat.p99)}
      </span>
    </div>
  );
}
