"use client";

import { useEffect, useRef, useState } from "react";
import { Layers } from "lucide-react";
import { Panel, Pill } from "./ui/panel";
import { metaFor, type Snapshot } from "@/lib/types";

// The C++ micro-batcher saturation over time, as small multiples — one tiny row
// per server with two sparklines: queue depth (how deep the bounded batcher
// queue is sitting) and average batch size (how many requests each ONNX run
// coalesces). These are the two numbers that say whether the serving path is
// keeping up or backing up. queue_depth and batch_avg are point-in-time per
// company, so we accumulate a short rolling history here just like the live
// time-series charts do.

const MAX_POINTS = 60; // ~1 min of history at 1 poll/sec

// per-server rolling history of the two saturation signals
interface SatHistory {
  queue: number[];
  batch: number[];
}

export function SaturationPanel({ snapshot }: { snapshot: Snapshot | null }) {
  const [hist, setHist] = useState<Record<string, SatHistory>>({});
  const lastKey = useRef<number>(-1);

  const ids = snapshot?.order?.length
    ? snapshot.order
    : snapshot
    ? Object.keys(snapshot.companies).sort()
    : [];

  // on each new snapshot push the current queue_depth / batch_avg for every
  // server onto its rolling history. guard against the same snapshot being
  // appended twice by keying on the auctions counter.
  useEffect(() => {
    if (!snapshot) return;
    const key = snapshot.auctions;
    if (key === lastKey.current) return;
    lastKey.current = key;
    setHist((prev) => {
      const next: Record<string, SatHistory> = {};
      for (const id of Object.keys(snapshot.companies)) {
        const c = snapshot.companies[id];
        const old = prev[id] ?? { queue: [], batch: [] };
        const q = [...old.queue, c.queue_depth ?? 0];
        const b = [...old.batch, c.batch_avg ?? 0];
        next[id] = {
          queue: q.length > MAX_POINTS ? q.slice(-MAX_POINTS) : q,
          batch: b.length > MAX_POINTS ? b.slice(-MAX_POINTS) : b,
        };
      }
      return next;
    });
  }, [snapshot]);

  return (
    <Panel
      icon={<Layers className="h-4 w-4" />}
      title="Micro-batcher saturation"
      subtitle="queue depth and batch size per server"
      info="Each server coalesces concurrent requests into one ONNX call. Avg batch size is ~1 at rest (no need to batch) and climbs under load (here ~8 in the max-capacity run); queue depth shows requests waiting for the next batch."
      right={<Pill tone="info">cpp batcher</Pill>}
    >
      {ids.length === 0 ? (
        <div className="h-32 grid place-items-center text-[11px] text-muted-foreground">
          waiting for scrape…
        </div>
      ) : (
        <div className="space-y-3">
          {/* column header */}
          <div className="grid grid-cols-[110px_1fr_1fr] gap-3 px-1 text-[9px] uppercase tracking-[0.1em] text-label">
            <span>server</span>
            <span>queue depth</span>
            <span>avg batch size</span>
          </div>
          {ids.map((id) => {
            const c = snapshot!.companies[id];
            const m = metaFor(id);
            const h = hist[id] ?? { queue: [], batch: [] };
            return (
              <div
                key={id}
                className="grid grid-cols-[110px_1fr_1fr] gap-3 items-center border-b border-hairline/60 pb-2.5"
              >
                <span className="text-[11px] font-medium flex items-center gap-1.5 mono truncate">
                  <span className="h-2 w-2 rounded-sm shrink-0" style={{ background: m.color }} />
                  {m.short.toLowerCase()}
                </span>
                <Spark series={h.queue} current={c.queue_depth ?? 0} color={m.color} digits={0} />
                <Spark series={h.batch} current={c.batch_avg ?? 0} color={m.color} digits={1} />
              </div>
            );
          })}
        </div>
      )}
    </Panel>
  );
}

// Inline-SVG sparkline with the current value pinned on the right: a thin line
// over a faint fill.
function Spark({
  series,
  current,
  color,
  digits,
}: {
  series: number[];
  current: number;
  color: string;
  digits: number;
}) {
  const W = 100;
  const H = 28;
  let max = 1;
  for (let i = 0; i < series.length; i++) {
    if (series[i] > max) max = series[i];
  }

  // build the line and the closed fill area in viewBox space
  const pts: string[] = [];
  const n = series.length;
  for (let i = 0; i < n; i++) {
    const x = n <= 1 ? W : (i / (n - 1)) * W;
    const y = H - (series[i] / max) * (H - 2) - 1;
    pts.push(`${x.toFixed(2)},${y.toFixed(2)}`);
  }
  const line = pts.join(" ");
  const area = n > 1 ? `0,${H} ${line} ${W},${H}` : "";

  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 min-w-0 h-7 rounded-sm bg-card-2/60 border border-hairline overflow-hidden">
        {n < 2 ? (
          <div className="h-full grid place-items-center text-[9px] mono text-label">
            collecting…
          </div>
        ) : (
          <svg
            className="w-full h-full"
            viewBox={`0 0 ${W} ${H}`}
            preserveAspectRatio="none"
          >
            <polygon points={area} fill={color} fillOpacity={0.12} />
            <polyline
              points={line}
              fill="none"
              stroke={color}
              strokeWidth={1}
              strokeOpacity={0.85}
              vectorEffect="non-scaling-stroke"
            />
          </svg>
        )}
      </div>
      <span className="w-10 shrink-0 text-right text-[10.5px] tnum mono" style={{ color }}>
        {current.toFixed(digits)}
      </span>
    </div>
  );
}
