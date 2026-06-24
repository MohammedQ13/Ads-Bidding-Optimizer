"use client";

import { useEffect, useRef, useState } from "react";
import { Activity } from "lucide-react";
import { Panel, Pill } from "./ui/panel";
import type { Snapshot } from "@/lib/types";
import { us } from "@/lib/utils";

// Latency heatmap: x-axis is time (one column per poll), y-axis is latency
// buckets, cell intensity is the request count in that bucket at that poll.

// Bucket edges in microseconds. rtt_us is round-trip µs; the fleet mostly sits
// in the sub-2ms range (p50 ~0.9ms), so the bands spread that range across rows.
// The top row is an open-ended "2.5ms+" catch-all for slow spikes under load,
// so nothing is dropped even when p99 climbs to a few ms.
const EDGES_US = [0, 400, 600, 800, 1000, 1200, 1500, 2500, Infinity];

// human labels for each row, top (slowest) to bottom (fastest) when rendered
const ROW_LABELS = [
  "2.5ms+",
  "1.5-2.5",
  "1.2-1.5",
  "1.0-1.2",
  "0.8-1.0",
  "0.6-0.8",
  "0.4-0.6",
  "<0.4ms",
];

const MAX_COLS = 50; // rolling window of recent polls
const NBINS = EDGES_US.length - 1;

// one heatmap column: per-bucket request counts plus p50/p99 of that poll
interface Column {
  key: number; // the snapshot.auctions value that produced this column
  counts: number[]; // length NBINS, index 0 = fastest bucket
  p50: number;
  p99: number;
  total: number;
}

// drop a single rtt_us sample into its bucket index (0 = fastest)
function bucketOf(rtt: number): number {
  for (let i = 0; i < NBINS; i++) {
    if (rtt >= EDGES_US[i] && rtt < EDGES_US[i + 1]) return i;
  }
  return NBINS - 1;
}

// build one column from the current events ring
function columnFrom(snapshot: Snapshot): Column {
  const counts = new Array(NBINS).fill(0);
  const samples: number[] = [];
  const events = snapshot.events ?? [];
  for (let e = 0; e < events.length; e++) {
    const bids = events[e].bids ?? [];
    for (let b = 0; b < bids.length; b++) {
      const bid = bids[b];
      // count real served requests only: skip paced (no call) and missing rtt
      if (bid.paced || !(bid.rtt_us > 0)) continue;
      counts[bucketOf(bid.rtt_us)]++;
      samples.push(bid.rtt_us);
    }
  }
  let p50 = 0;
  let p99 = 0;
  if (samples.length > 0) {
    samples.sort((a, b) => a - b);
    p50 = samples[Math.min(samples.length - 1, Math.floor(0.5 * samples.length))];
    p99 = samples[Math.min(samples.length - 1, Math.floor(0.99 * samples.length))];
  }
  return { key: snapshot.auctions, counts, p50, p99, total: samples.length };
}

export function LatencyHeatmap({ snapshot }: { snapshot: Snapshot | null }) {
  const [cols, setCols] = useState<Column[]>([]);
  const lastKey = useRef<number>(-1);

  // on each fresh snapshot, histogram the current events ring into one new
  // column and push it onto the rolling buffer. guard against the same
  // snapshot (same auctions count) being counted twice across re-renders.
  useEffect(() => {
    if (!snapshot) return;
    const key = snapshot.auctions;
    if (key === lastKey.current) return;
    lastKey.current = key;
    const col = columnFrom(snapshot);
    setCols((prev) => {
      const next = [...prev, col];
      return next.length > MAX_COLS ? next.slice(-MAX_COLS) : next;
    });
  }, [snapshot]);

  // peak count across every cell, for normalizing color intensity
  let peak = 1;
  for (let c = 0; c < cols.length; c++) {
    for (let r = 0; r < NBINS; r++) {
      if (cols[c].counts[r] > peak) peak = cols[c].counts[r];
    }
  }

  const hasData = cols.length > 0 && cols.some((c) => c.total > 0);

  return (
    <Panel
      icon={<Activity className="h-4 w-4" />}
      title="Latency distribution over time"
      subtitle="round-trip latency per request, by bucket"
      info="Round-trip latency over time: each column is one poll, rows are latency bands (fast at the bottom), and a brighter cell means more requests landed in that band. The lines trace p50 (solid) and p99 (dashed)."
      right={<Pill tone="info">heatmap · {cols.length} polls</Pill>}
    >
      {!hasData ? (
        <div className="h-44 grid place-items-center text-[11px] text-muted-foreground">
          collecting latency samples…
        </div>
      ) : (
        <Grid cols={cols} peak={peak} />
      )}
    </Panel>
  );
}

// The heatmap grid itself: a fixed label gutter on the left, then a flex row of
// thin columns. Each column is a vertical stack of NBINS cells, slowest bucket
// on top. We overlay thin p50 / p99 polylines across the columns in SVG.
function Grid({ cols, peak }: { cols: Column[]; peak: number }) {
  // row order for rendering: slowest (highest index) at the top
  const rows: number[] = [];
  for (let r = NBINS - 1; r >= 0; r--) rows.push(r);

  // map a microsecond value to a vertical fraction (0 = top, 1 = bottom) using
  // the same row banding the cells use, so the percentile lines line up. each
  // row owns an equal vertical slice; we place the value at its band center.
  const yFracOf = (v: number) => {
    if (!(v > 0)) return null;
    const bi = bucketOf(v); // 0 = fastest (bottom)
    const rowFromTop = NBINS - 1 - bi; // 0 = top row
    return (rowFromTop + 0.5) / NBINS;
  };

  const n = cols.length;
  const xFracOf = (i: number) => (n <= 1 ? 1 : (i + 0.5) / n);

  // build the polyline point strings in a 0..100 viewBox space
  const p50Pts: string[] = [];
  const p99Pts: string[] = [];
  for (let i = 0; i < n; i++) {
    const x = xFracOf(i) * 100;
    const y50 = yFracOf(cols[i].p50);
    const y99 = yFracOf(cols[i].p99);
    if (y50 != null) p50Pts.push(`${x.toFixed(2)},${(y50 * 100).toFixed(2)}`);
    if (y99 != null) p99Pts.push(`${x.toFixed(2)},${(y99 * 100).toFixed(2)}`);
  }

  return (
    <div>
      <div className="flex items-stretch gap-2">
        {/* y-axis labels */}
        <div className="flex flex-col justify-between text-right text-[8.5px] mono tnum text-label py-px w-12 shrink-0">
          {rows.map((r) => (
            <div key={r} className="leading-none h-[18px] flex items-center justify-end">
              {ROW_LABELS[NBINS - 1 - r]}
            </div>
          ))}
        </div>

        {/* the cell grid + percentile overlay */}
        <div className="relative flex-1 min-w-0">
          <div className="flex gap-px h-full border border-hairline rounded-sm overflow-hidden">
            {cols.map((col, ci) => (
              <div key={`${col.key}-${ci}`} className="flex-1 flex flex-col gap-px min-w-0">
                {rows.map((r) => {
                  const count = col.counts[r];
                  return (
                    <div
                      key={r}
                      className="h-[18px] relative group"
                      style={{ background: cellColor(count, peak) }}
                    >
                      {count > 0 && (
                        <span className="pointer-events-none absolute z-10 left-1/2 -translate-x-1/2 -top-6 whitespace-nowrap rounded bg-card border border-border px-1 py-0.5 text-[9px] mono opacity-0 group-hover:opacity-100">
                          {ROW_LABELS[NBINS - 1 - r]} · {count}
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>

          {/* p50 / p99 lines drawn across the columns */}
          <svg
            className="absolute inset-0 w-full h-full pointer-events-none"
            viewBox="0 0 100 100"
            preserveAspectRatio="none"
          >
            {p50Pts.length > 1 && (
              <polyline
                points={p50Pts.join(" ")}
                fill="none"
                stroke="var(--muted-foreground)"
                strokeWidth={0.6}
                strokeOpacity={0.7}
                vectorEffect="non-scaling-stroke"
              />
            )}
            {p99Pts.length > 1 && (
              <polyline
                points={p99Pts.join(" ")}
                fill="none"
                stroke="var(--warning)"
                strokeWidth={0.8}
                strokeOpacity={0.85}
                strokeDasharray="2 1.5"
                vectorEffect="non-scaling-stroke"
              />
            )}
          </svg>
        </div>
      </div>

      {/* x-axis: time, most recent on the right */}
      <div className="flex items-center gap-2 mt-1.5">
        <div className="w-12 shrink-0" />
        <div className="flex-1 flex justify-between text-[8.5px] mono text-label">
          <span>oldest</span>
          <span>recent →</span>
        </div>
      </div>

      {/* footer legend: the sequential ramp + percentile line key */}
      <div className="flex items-center justify-between mt-2 pt-2 border-t border-hairline text-[9px] mono text-label">
        <div className="flex items-center gap-1.5">
          <span className="normal-case">low</span>
          <div className="flex">
            {[0, 0.2, 0.4, 0.6, 0.8, 1].map((f) => (
              <span
                key={f}
                className="h-2.5 w-4 first:rounded-l-sm last:rounded-r-sm"
                style={{ background: cellColor(f * peak, peak) }}
              />
            ))}
          </div>
          <span className="normal-case">high (req/poll)</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1">
            <span className="inline-block w-3 border-t" style={{ borderColor: "var(--muted-foreground)" }} />
            p50
          </span>
          <span className="flex items-center gap-1">
            <span
              className="inline-block w-3 border-t border-dashed"
              style={{ borderColor: "var(--warning)" }}
            />
            p99
          </span>
        </div>
      </div>
    </div>
  );
}

// single-hue sequential ramp; opacity encodes count
function cellColor(count: number, peak: number): string {
  if (count <= 0) return "var(--card-2)";
  const t = Math.min(1, count / peak);
  // gamma-lift so low counts stay visible against the dark canvas
  const a = 0.12 + Math.pow(t, 0.7) * 0.78;
  return `color-mix(in srgb, var(--primary) ${(a * 100).toFixed(0)}%, transparent)`;
}
