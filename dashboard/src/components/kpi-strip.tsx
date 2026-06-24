"use client";

import { Sparkline } from "./sparkline";
import type { HistPoint } from "@/lib/use-scoreboard";
import type { Snapshot } from "@/lib/types";
import { compact, fmt, pct, us } from "@/lib/utils";

// The headline metric strip: one number per system concern, each with a recent
// trend sparkline pulled from the rolling history.
export function KpiStrip({
  snapshot,
  history,
}: {
  snapshot: Snapshot | null;
  history: HistPoint[];
}) {
  const ids = snapshot ? Object.keys(snapshot.companies) : [];

  let totalProfit = 0;
  let fleetServed = 0;
  let modelVersion = 0;
  let infSum = 0;
  let infN = 0;
  let fallbackSum = 0;
  for (const id of ids) {
    const c = snapshot!.companies[id];
    totalProfit += c.profit;
    fleetServed += c.served_per_sec ?? 0;
    modelVersion = Math.max(modelVersion, c.model_version);
    if (c.inference_us.p50 > 0) {
      infSum += c.inference_us.p50;
      infN++;
    }
    fallbackSum += c.fallback_rate;
  }
  const avgInf = infN > 0 ? infSum / infN : 0;
  const avgFallback = ids.length > 0 ? fallbackSum / ids.length : 0;

  // color carries meaning: accent on throughput, green/amber on fallback rate,
  // everything else neutral
  const tiles = [
    {
      label: "Exchange rate",
      value: snapshot ? `${compact(snapshot.auctions_per_sec)}` : "—",
      unit: "auctions/s",
      series: history.map((h) => h.aps),
      color: "#8e8e9e",
    },
    {
      label: "Fleet p99 RTT",
      value: snapshot ? us(maxP99(snapshot)) : "—",
      unit: "round-trip",
      series: history.map((h) => h.fleetP99),
      color: "#8e8e9e",
    },
    {
      label: "Inference p50",
      value: us(avgInf),
      unit: "onnx run",
      series: history.map((h) => h.inferenceP50),
      color: "#8e8e9e",
    },
    {
      label: "Clearing price",
      value: snapshot ? `${Math.round(snapshot.last_clearing_price)}` : "—",
      unit: "fen",
      series: history.map((h) => h.clearing),
      color: "#8e8e9e",
    },
    {
      label: "Fallback rate",
      value: pct(avgFallback),
      unit: "model bypass",
      series: history.map((h) => h.fallbackRate),
      color: avgFallback > 0.05 ? "#e0a000" : "#6ccf8e",
    },
    {
      label: "Total profit",
      value: snapshot ? fmt(totalProfit) : "—",
      unit: "fen, all strategies",
      series: ids.length ? history.map((h) => ids.reduce((s, id) => s + (h[id] ?? 0), 0)) : [],
      color: "#ccccdc",
    },
    {
      label: "Fleet req/s",
      value: snapshot ? `${compact(fleetServed)}` : "—",
      unit: "served, all servers",
      series: history.map((h) => h.fleetServed),
      color: "#3d71d9",
      accent: true,
    },
    {
      label: "Model version",
      value: modelVersion ? `v${modelVersion}` : "v1",
      unit: "hot-reloads",
      series: [],
      color: "#8e8e9e",
    },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-8 gap-2.5">
      {tiles.map((t) => (
        <div
          key={t.label}
          className="bg-card border border-border rounded-lg p-3 animate-enter relative overflow-hidden"
        >
          <p className="text-[9.5px] uppercase tracking-[0.1em] text-label">
            {t.label}
          </p>
          <p
            className="text-[21px] font-semibold tnum mono leading-none mt-1.5"
            style={{ color: t.color }}
          >
            {t.value}
          </p>
          <p className="text-[9.5px] text-muted-foreground mt-1">{t.unit}</p>
          {t.series.length > 1 && (
            <Sparkline
              data={t.series}
              color={t.color}
              className="h-6 w-full mt-1.5 opacity-80"
            />
          )}
        </div>
      ))}
    </div>
  );
}

function maxP99(s: Snapshot): number {
  let m = 0;
  for (const id of Object.keys(s.companies)) {
    m = Math.max(m, s.companies[id].rtt_us.p99);
  }
  return m;
}
