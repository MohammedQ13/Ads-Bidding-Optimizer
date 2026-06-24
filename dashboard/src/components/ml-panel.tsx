"use client";

import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { Brain } from "lucide-react";
import { Panel, Pill } from "./ui/panel";
import { Histogram } from "./ui/histogram";
import { metaFor, type Snapshot } from "@/lib/types";
import { fmt, pct } from "@/lib/utils";

// Model output per bid: the C++ servers return win_prob and expected_profit,
// the engine forwards them. Plots the bid -> P(win) scatter, the win-prob
// histogram, predicted vs realized win rate, and expected vs realized profit.
export function MlPanel({ snapshot }: { snapshot: Snapshot | null }) {
  const ids = snapshot?.order?.length
    ? snapshot.order
    : snapshot
    ? Object.keys(snapshot.companies).sort()
    : [];

  // scatter: (bid, win_prob) for every recent bid, grouped by company
  const seriesByCompany: Record<string, { x: number; y: number }[]> = {};
  for (const ev of snapshot?.events ?? []) {
    for (const b of ev.bids) {
      if (!b.ok || b.paced || b.bid <= 0) continue;
      (seriesByCompany[b.id] ??= []).push({ x: b.bid, y: b.win_prob });
    }
  }

  return (
    <Panel
      icon={<Brain className="h-4 w-4" />}
      title="Model predictions"
      subtitle="discrete-bin clearing-price distribution"
      info="What the model outputs: predicted win probability for each bid, the spread of those probabilities, calibration (predicted vs actually-realized win rate), and expected vs booked profit. The pred-vs-real gap shows the model meeting this synthetic market."
      right={<Pill tone="info">model output</Pill>}
    >
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* learned price -> P(win) curve */}
        <Tile
          title="P(win) by bid"
          hint="one dot per recent bid"
        >
          <div className="h-[170px]">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 8, right: 8, bottom: 4, left: -8 }}>
                <CartesianGrid stroke="rgba(204,204,220,0.07)" strokeDasharray="2 4" />
                <XAxis
                  type="number"
                  dataKey="x"
                  name="bid"
                  tick={{ fontSize: 9, fill: "#6e7079" }}
                  tickLine={false}
                  axisLine={false}
                  unit=" fen"
                  domain={[0, "dataMax"]}
                />
                <YAxis
                  type="number"
                  dataKey="y"
                  name="P(win)"
                  tick={{ fontSize: 9, fill: "#6e7079" }}
                  tickLine={false}
                  axisLine={false}
                  domain={[0, 1]}
                  tickFormatter={(v) => `${Math.round((v as number) * 100)}%`}
                  width={40}
                />
                <ZAxis range={[14, 14]} />
                <Tooltip content={<ScatterTip />} cursor={{ strokeDasharray: "3 3" }} />
                {ids.map((id) => (
                  <Scatter
                    key={id}
                    data={seriesByCompany[id] ?? []}
                    fill={metaFor(id).color}
                    fillOpacity={0.55}
                    isAnimationActive={false}
                  />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </Tile>

        {/* win-probability confidence distribution */}
        <Tile
          title="P(win) distribution"
          hint="win-prob across recent bids"
        >
          <Histogram
            buckets={(snapshot?.win_prob_hist ?? []).map((b) => ({
              lo: Math.round(b.lo * 100),
              hi: Math.round(b.hi * 100),
              count: b.count,
            }))}
            color="var(--primary)"
            unit="%"
            height={150}
            openTop={false}
          />
        </Tile>

        {/* calibration: predicted vs realized */}
        <Tile
          title="Calibration"
          hint="predicted vs realized win rate"
        >
          <div className="space-y-2.5 pt-1">
            {ids.map((id) => {
              const c = snapshot!.companies[id];
              return (
                <PairRow
                  key={id}
                  label={metaFor(id).name}
                  color={metaFor(id).color}
                  a={c.avg_win_prob}
                  b={c.win_rate}
                  fmt={pct}
                  aLabel="pred"
                  bLabel="real"
                />
              );
            })}
          </div>
        </Tile>

        {/* expected vs realized profit */}
        <Tile
          title="Expected vs realized profit"
          hint="Σ model E[profit] over all bids vs booked profit on wins"
        >
          <div className="space-y-2.5 pt-1">
            {ids.map((id) => {
              const c = snapshot!.companies[id];
              // scale on magnitude so a negative (overpaying) strategy still
              // draws a visible bar instead of collapsing to zero width
              const max = Math.max(1, Math.abs(c.expected_profit), Math.abs(c.profit));
              return (
                <PairRow
                  key={id}
                  label={metaFor(id).name}
                  color={metaFor(id).color}
                  a={c.expected_profit}
                  b={c.profit}
                  max={max}
                  fmt={fmt}
                  aLabel="exp"
                  bLabel="real"
                />
              );
            })}
          </div>
        </Tile>
      </div>
    </Panel>
  );
}

function Tile({
  title,
  hint,
  children,
}: {
  title: string;
  hint: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-md border border-border bg-card-2 p-3">
      <p className="text-[11px] font-semibold">{title}</p>
      <p className="text-[9.5px] text-muted-foreground mb-2">{hint}</p>
      {children}
    </div>
  );
}

function PairRow({
  label,
  color,
  a,
  b,
  max,
  fmt: f,
  aLabel,
  bLabel,
}: {
  label: string;
  color: string;
  a: number;
  b: number;
  max?: number;
  fmt: (n: number) => string;
  aLabel: string;
  bLabel: string;
}) {
  const top = max ?? 1;
  // width from magnitude; a negative value (e.g. a strategy overpaying past V)
  // renders in the destructive red so it reads as a real loss, not a missing bar
  const wA = `${Math.min(100, (Math.abs(a) / top) * 100)}%`;
  const wB = `${Math.min(100, (Math.abs(b) / top) * 100)}%`;
  const colA = a < 0 ? "var(--destructive)" : color;
  const colB = b < 0 ? "var(--destructive)" : `${color}77`;
  return (
    <div>
      <div className="flex items-center justify-between text-[10px] mb-1">
        <span className="font-medium">{label}</span>
        <span className="mono text-muted-foreground tnum">
          {aLabel} {f(a)} · {bLabel} {f(b)}
        </span>
      </div>
      <div className="space-y-[3px]">
        <div className="h-2 rounded-sm bg-muted/40 overflow-hidden">
          <div className="h-full rounded-sm" style={{ width: wA, background: colA }} />
        </div>
        <div className="h-2 rounded-sm bg-muted/40 overflow-hidden">
          <div className="h-full rounded-sm" style={{ width: wB, background: colB }} />
        </div>
      </div>
    </div>
  );
}

function ScatterTip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload: { x: number; y: number } }>;
}) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="bg-card border border-border rounded-md px-2 py-1 text-[10px] mono shadow-raised">
      bid {Math.round(p.x)} fen · P(win) {Math.round(p.y * 100)}%
    </div>
  );
}
