"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  YAxis,
} from "recharts";
import { metaFor } from "@/lib/types";
import type { HistPoint } from "@/lib/use-scoreboard";
import { fmt } from "@/lib/utils";

export function ProfitRaceChart({
  history,
  companies,
}: {
  history: HistPoint[];
  companies: string[];
}) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={history} margin={{ top: 6, right: 6, bottom: 0, left: 0 }}>
        <defs>
          {companies.map((id) => {
            const color = metaFor(id).color;
            return (
              <linearGradient key={id} id={`grad-${id}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={color} stopOpacity={0.28} />
                <stop offset="100%" stopColor={color} stopOpacity={0} />
              </linearGradient>
            );
          })}
        </defs>
        <CartesianGrid
          stroke="rgba(204,204,220,0.07)"
          strokeDasharray="2 4"
          vertical={false}
        />
        <YAxis
          width={44}
          tick={{ fontSize: 10, fill: "#6e7079" }}
          tickLine={false}
          axisLine={false}
          tickFormatter={(v) => fmt(v as number)}
        />
        <Tooltip
          content={<RaceTooltip />}
          isAnimationActive={false}
          cursor={{ stroke: "rgba(204,204,220,0.18)", strokeWidth: 1 }}
        />
        {companies.map((id) => (
          <Area
            key={id}
            type="monotone"
            dataKey={id}
            stroke={metaFor(id).color}
            strokeWidth={2}
            fill={`url(#grad-${id})`}
            dot={false}
            isAnimationActive={false}
            connectNulls
          />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  );
}

function RaceTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ dataKey: string; value: number; color: string }>;
}) {
  if (!active || !payload?.length) return null;
  const sorted = [...payload].sort((a, b) => b.value - a.value);
  return (
    <div className="bg-card border border-border rounded-md shadow-raised px-3 py-2 text-[11px] space-y-1">
      {sorted.map((p) => (
        <div key={p.dataKey} className="flex items-center justify-between gap-5 tnum mono">
          <span className="flex items-center gap-1.5 text-muted-foreground">
            <span className="h-2 w-2 rounded-full" style={{ background: p.color }} />
            {metaFor(p.dataKey).name}
          </span>
          <span className="font-medium text-foreground">{fmt(p.value)}</span>
        </div>
      ))}
    </div>
  );
}
