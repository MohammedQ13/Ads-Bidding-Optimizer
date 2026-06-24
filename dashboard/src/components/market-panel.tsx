"use client";

import { Store } from "lucide-react";
import { Panel, Pill, Bars } from "./ui/panel";
import { Histogram } from "./ui/histogram";
import type { Snapshot } from "@/lib/types";
import { fmt, pct, compact } from "@/lib/utils";

// The synthetic market the strategies bid into: competitor archetypes, the
// field and clearing-price distributions, and the impression mix (hour of day,
// exchange, slot size, floor presence).
export function MarketPanel({ snapshot }: { snapshot: Snapshot | null }) {
  const arch = snapshot?.competitor_archetypes ?? [];
  // muted slate -> mauve ramp: the competitors are market context, kept desaturated
  // so they stay subordinate to the vivid company series
  const archColors = [
    "#6e7bb0", "#7c84ac", "#8a83a6", "#9882a2",
    "#6f88a4", "#7e86a0", "#8d849c", "#9b8299",
  ];

  const byHour = (snapshot?.impressions.by_hour ?? []).map((count, h) => ({
    lo: h,
    hi: h + 1,
    count,
  }));

  const byExchange = Object.entries(snapshot?.impressions.by_exchange ?? {})
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => ({ label: `ADX ${k}`, value: v, color: "var(--primary)" }));

  const bySlot = Object.entries(snapshot?.impressions.by_slot ?? {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6)
    .map(([k, v]) => ({ label: k, value: v, color: "var(--company-a)" }));

  return (
    <Panel
      icon={<Store className="h-4 w-4" />}
      title="Market & impressions"
      subtitle={`${snapshot?.num_competitors ?? 0} synthetic competitors per auction`}
      info="The synthetic market the strategies bid into: competitor archetypes (whale down to bargain) by their average bid, the field and clearing-price distributions, and the mix of impressions by hour, exchange and slot size."
      right={
        <Pill tone="muted">
          floor present {pct(snapshot?.impressions.floor_rate ?? 0)}
        </Pill>
      }
    >
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Tile title="Competitor archetypes" hint="avg bid per archetype (every archetype bids every auction)">
          <Bars
            items={arch.map((a, i) => ({
              label: a.label,
              value: a.avg_bid,
              color: archColors[i % archColors.length],
              sub: `${compact(a.count)} bids`,
            }))}
            format={(n) => `${fmt(n)} fen`}
          />
        </Tile>

        <Tile title="Competitor bid distribution" hint="fen">
          <Histogram
            buckets={snapshot?.histograms?.competitors ?? []}
            color="var(--company-d)"
            unit=""
            height={150}
          />
        </Tile>

        <Tile title="Clearing price vs our bids" hint="fen · each series scaled to its own peak">
          <Histogram
            buckets={snapshot?.histograms?.clearing ?? []}
            overlay={snapshot?.histograms?.bids ?? []}
            color="var(--muted-foreground)"
            overlayColor="var(--primary)"
            unit=""
            height={150}
            normalizeSeparately
          />
          <div className="flex items-center gap-4 mt-2 text-[9.5px] mono text-muted-foreground">
            <span className="flex items-center gap-1"><i className="h-2 w-2 rounded-sm inline-block" style={{ background: "var(--muted-foreground)" }} /> clearing</span>
            <span className="flex items-center gap-1"><i className="h-2 w-2 rounded-sm inline-block" style={{ background: "var(--primary)" }} /> our bids</span>
          </div>
        </Tile>

        <Tile title="Impressions by hour" hint="count by hour">
          <Histogram buckets={byHour} color="var(--muted-foreground)" unit="h" height={150} />
        </Tile>

        <Tile title="By ad exchange" hint="share by exchange">
          <Bars items={byExchange} format={fmt} />
        </Tile>

        <Tile title="By slot size" hint="top slot sizes">
          <Bars items={bySlot} format={fmt} />
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
      <p className="text-[9.5px] text-muted-foreground mb-2.5">{hint}</p>
      {children}
    </div>
  );
}
