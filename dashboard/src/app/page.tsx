"use client";

import { useState } from "react";
import { Moon, TrendingUp } from "lucide-react";
import { useScoreboard } from "@/lib/use-scoreboard";
import { CommandBar } from "@/components/command-bar";
import { Topology } from "@/components/topology";
import { KpiStrip } from "@/components/kpi-strip";
import { TraceWaterfall } from "@/components/trace-waterfall";
import { LatencyPanel } from "@/components/latency-panel";
import { LatencyHeatmap } from "@/components/latency-heatmap";
import { SaturationPanel } from "@/components/saturation-panel";
import { ProfitRaceChart } from "@/components/profit-race-chart";
import { CompanyCard } from "@/components/company-card";
import { MlPanel } from "@/components/ml-panel";
import { MarketPanel } from "@/components/market-panel";
import { CppInternals } from "@/components/cpp-internals";
import { AuctionStream } from "@/components/auction-stream";
import { RetrainerPanel } from "@/components/retrainer-panel";
import { Panel } from "@/components/ui/panel";
import { Card } from "@/components/ui/card";
import { metaFor } from "@/lib/types";

export default function Page() {
  const [mode, setMode] = useState<"live" | "maxcap">("live");
  const { snapshot, history, state } = useScoreboard(mode);

  const companies = snapshot
    ? snapshot.order?.length
      ? snapshot.order
      : Object.keys(snapshot.companies).sort()
    : [];

  let leader = "";
  let leaderProfit = -Infinity;
  let dailyBudget = 1;
  for (const id of companies) {
    const c = snapshot!.companies[id];
    if (c.profit > leaderProfit) {
      leaderProfit = c.profit;
      leader = id;
    }
    if (c.budget > dailyBudget) dailyBudget = c.budget;
  }

  const seriesFor = (id: string) =>
    history.slice(-40).map((p) => (p[id] ?? 0) as number);

  const showOffline = state === "offline" && !snapshot;

  return (
    <div className="min-h-screen">
      <CommandBar snapshot={snapshot} state={state} mode={mode} setMode={setMode} />

      <main className="mx-auto max-w-[1500px] px-4 lg:px-6 py-5 space-y-4">
        <CapabilityStrip mode={mode} />
        {showOffline ? (
          <OfflineHero />
        ) : (
          <>
            {/* system panels */}
            <Topology snapshot={snapshot} />

            <KpiStrip snapshot={snapshot} history={history} />

            <TraceWaterfall snapshot={snapshot} />

            <LatencyPanel snapshot={snapshot} />

            <LatencyHeatmap key={`heatmap-${mode}`} snapshot={snapshot} />

            {/* economics panels */}
            <Panel
              icon={<TrendingUp className="h-4 w-4" />}
              title="Cumulative profit"
              subtitle="realized profit per strategy"
              info="Realized profit per strategy over time. First-price auction, so profit = impression value minus the winning bid; a strategy that overbids (e.g. the 1.2x aggressive one) can book a real loss and go negative."
              right={<Legend companies={companies} />}
            >
              <div className="h-[300px] w-full">
                {history.length > 1 ? (
                  <ProfitRaceChart history={history} companies={companies} />
                ) : (
                  <div className="h-full grid place-items-center text-[12px] text-muted-foreground">
                    {state === "live" ? "collecting data…" : "connecting to the engine…"}
                  </div>
                )}
              </div>
            </Panel>

            <section className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
              {companies.length > 0
                ? companies.map((id) => (
                    <CompanyCard
                      key={id}
                      id={id}
                      stats={snapshot!.companies[id]}
                      series={seriesFor(id)}
                      dailyBudget={dailyBudget}
                      leader={id === leader && companies.length > 1}
                    />
                  ))
                : Array.from({ length: 4 }).map((_, i) => (
                    <Card key={i} className="p-4 h-[230px]">
                      <div className="h-3 w-20 rounded bg-muted animate-pulse" />
                      <div className="h-7 w-28 rounded bg-muted animate-pulse mt-4" />
                      <div className="h-8 w-full rounded bg-muted animate-pulse mt-6" />
                    </Card>
                  ))}
            </section>

            <AuctionStream snapshot={snapshot} />

            <MlPanel snapshot={snapshot} />

            <MarketPanel snapshot={snapshot} />

            <SaturationPanel key={`saturation-${mode}`} snapshot={snapshot} />

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <CppInternals snapshot={snapshot} />
              <RetrainerPanel snapshot={snapshot} />
            </div>
          </>
        )}
      </main>
    </div>
  );
}

// Always-visible strip of benchmarked ceilings (one 16-vCPU node). Measured
// figures, not live telemetry.
function CapabilityStrip({ mode }: { mode: "live" | "maxcap" }) {
  return (
    <div className="rounded-md border border-hairline bg-card-2 px-3 py-2 flex flex-wrap items-center gap-x-4 gap-y-1 mono text-[10.5px]">
      <span className="label-micro text-label">benchmarked ceiling · 1 node</span>
      <Cap k="e2e serving" v="~56k req/s @ p99 2.4ms" />
      <Cap k="peak" v="~79k req/s" />
      <Cap k="compute" v="~965k bids/sec (~1µs/bid)" />
      <Cap k="scale" v="stateless, scales with replicas" />
      <span className="text-muted-foreground/70">
        {mode === "maxcap"
          ? "recorded run with the gRPC fleet under heavy concurrent load (micro-batcher engaging, elevated tail latency)"
          : "live market below is paced ~200/s for readability"}
      </span>
    </div>
  );
}

function Cap({ k, v }: { k: string; v: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="text-label">{k}</span>
      <span className="text-foreground tnum">{v}</span>
    </span>
  );
}

function Legend({ companies }: { companies: string[] }) {
  return (
    <div className="hidden sm:flex items-center gap-3">
      {companies.map((id) => {
        const m = metaFor(id);
        return (
          <span key={id} className="inline-flex items-center gap-1.5 text-[10.5px]">
            <span className="h-2 w-2 rounded-full" style={{ background: m.color }} />
            <span className="text-muted-foreground">{m.name}</span>
          </span>
        );
      })}
    </div>
  );
}

function OfflineHero() {
  return (
    <div className="grid place-items-center py-24 animate-enter">
      <Card className="p-10 max-w-md text-center">
        <span className="mx-auto grid h-12 w-12 place-items-center rounded-xl bg-muted text-muted-foreground">
          <Moon className="h-5 w-5" />
        </span>
        <h2 className="text-[16px] font-semibold mt-4">Control plane offline</h2>
        <p className="text-[12px] text-muted-foreground mt-2 leading-relaxed">
          The RTB backend isn&apos;t reachable right now. Start the stack (the four
          C++ bidders, the Go auction engine, and the retrainer) and this console
          connects automatically.
        </p>
        <p className="text-[11px] text-muted-foreground/70 mt-4 tnum mono">
          retrying every 1s…
        </p>
      </Card>
    </div>
  );
}
