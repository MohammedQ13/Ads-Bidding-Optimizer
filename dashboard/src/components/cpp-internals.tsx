"use client";

import { Server } from "lucide-react";
import { Panel, Pill } from "./ui/panel";
import { breakerLabel, metaFor, type Snapshot } from "@/lib/types";
import { compact, us } from "@/lib/utils";

// Per-server C++ internals scraped from Prometheus /metrics: micro-batcher
// queue depth, average batch size, dedup cache hits, circuit breaker state,
// cold-start vocab misses, and the loaded model version.
export function CppInternals({ snapshot }: { snapshot: Snapshot | null }) {
  const ids = snapshot?.order?.length
    ? snapshot.order
    : snapshot
    ? Object.keys(snapshot.companies).sort()
    : [];

  const cols: { key: string; label: string; hint: string }[] = [
    { key: "inf", label: "Inf p99", hint: "onnx" },
    { key: "queue", label: "Queue", hint: "batcher" },
    { key: "batch", label: "Batch", hint: "avg size" },
    { key: "cache", label: "Cache", hint: "dedup hits" },
    { key: "cold", label: "Cold", hint: "vocab miss" },
    { key: "model", label: "Model", hint: "loaded ver" },
    { key: "brk", label: "Breaker", hint: "state" },
  ];

  return (
    <Panel
      icon={<Server className="h-4 w-4" />}
      title="C++ serving internals"
      subtitle="scraped from /metrics"
      info="Live counters pulled from each C++ server's Prometheus /metrics page: ONNX inference p99, batcher queue/size, dedup cache hits, cold-start vocab misses, loaded model version, and circuit-breaker state."
      right={<Pill tone="info">prometheus scrape</Pill>}
      bodyClassName="p-0"
    >
      <div className="overflow-x-auto scroll-thin">
        <table className="w-full text-[11px] mono">
          <thead>
            <tr className="text-label border-b border-hairline">
              <th className="text-left font-medium px-4 py-2 sticky left-0 bg-card">Server</th>
              {cols.map((c) => (
                <th key={c.key} className="text-right font-medium px-3 py-2">
                  <div className="leading-none">{c.label}</div>
                  <div className="text-[8.5px] text-label/70 normal-case mt-0.5">{c.hint}</div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {ids.length === 0 && (
              <tr>
                <td colSpan={cols.length + 1} className="text-center text-muted-foreground py-8">
                  waiting for scrape…
                </td>
              </tr>
            )}
            {ids.map((id) => {
              const c = snapshot!.companies[id];
              const m = metaFor(id);
              const bk = breakerLabel(c.breaker_state);
              return (
                <tr key={id} className="border-b border-hairline/60 hover:bg-muted/30">
                  <td className="px-4 py-2 sticky left-0 bg-card">
                    <span className="flex items-center gap-2 font-sans font-medium">
                      <span className="h-2 w-2 rounded-full" style={{ background: m.color }} />
                      {m.name}
                    </span>
                  </td>
                  <td className="text-right px-3 py-2 tnum" style={{ color: m.color }}>
                    {us(c.inference_us.p99)}
                  </td>
                  <td className="text-right px-3 py-2 tnum">{c.queue_depth}</td>
                  <td className="text-right px-3 py-2 tnum">{c.batch_avg.toFixed(1)}</td>
                  <td className="text-right px-3 py-2 tnum">{compact(c.cache_hits)}</td>
                  <td className="text-right px-3 py-2 tnum">{compact(c.cold_start)}</td>
                  <td className="text-right px-3 py-2 tnum">v{c.model_version}</td>
                  <td className="text-right px-3 py-2">
                    <Pill tone={bk.tone}>{bk.label}</Pill>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}
