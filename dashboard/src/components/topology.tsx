"use client";

import { Boxes, Cpu, Network, RefreshCw } from "lucide-react";
import { Panel, Pill } from "./ui/panel";
import { breakerLabel, metaFor, type Snapshot } from "@/lib/types";
import { compact, us } from "@/lib/utils";

// Service map: the engine fans GetBid to the C++ fleet, outcomes flow back to the
// retrainer, and new models hot-swap in. Edges animate while traffic flows.
export function Topology({ snapshot }: { snapshot: Snapshot | null }) {
  const ids = snapshot?.order?.length
    ? snapshot.order
    : snapshot
    ? Object.keys(snapshot.companies).sort()
    : ["company-a", "company-b", "company-c", "company-d"];

  const retr = snapshot?.retrainer;
  const engineUp = !!snapshot;
  const aps = snapshot?.auctions_per_sec ?? 0;
  // real serving throughput across the fleet (counter-derived, includes load)
  let fleetServed = 0;
  if (snapshot) {
    for (const id of ids) fleetServed += snapshot.companies[id]?.served_per_sec ?? 0;
  }

  return (
    <Panel
      icon={<Network className="h-4 w-4" />}
      title="Service map"
      subtitle="gRPC fan-out, engine to fleet"
      info="The pipeline: the Go exchange fans each impression out to 4 C++ bidder servers over gRPC, collects their bids, settles the auction, and streams outcomes to the Python retrainer that hot-swaps new models."
      right={
        <Pill tone={engineUp ? "ok" : "bad"}>{engineUp ? "OPERATIONAL" : "DOWN"}</Pill>
      }
    >
      <div className="flex items-stretch gap-2 lg:gap-3">
        {/* Engine */}
        <Node
          className="w-[176px] shrink-0"
          icon={<Boxes className="h-4 w-4" />}
          title="auction-engine"
          tag="Go"
          health={engineUp ? "up" : "down"}
          rows={[
            ["auctions/s", snapshot ? `${compact(aps)}/s` : "—"],
            ["deadline", snapshot ? `${snapshot.deadline_ms}ms` : "—"],
            ["competitors", snapshot ? `${snapshot.num_competitors}/auc` : "—"],
            ["V", snapshot ? `${snapshot.impression_value} fen` : "—"],
          ]}
        />

        <Edge
          label="GetBid"
          sub={engineUp ? `${compact(aps)}/s ×${ids.length}` : "—"}
          live={engineUp && aps > 0}
        />

        {/* C++ fleet */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between mb-1.5 px-1">
            <span className="text-[10px] uppercase tracking-[0.1em] text-label flex items-center gap-1.5">
              <Cpu className="h-3.5 w-3.5 text-muted-foreground" /> cpp-bidder fleet
            </span>
            <span className="text-[9.5px] mono text-muted-foreground">
              {fleetServed > 0
                ? `serving ${compact(fleetServed)} req/s · shared model`
                : "shared model · per-strategy multiplier"}
            </span>
          </div>
          <div className="grid grid-cols-2 xl:grid-cols-4 gap-2">
            {ids.map((id) => {
              const c = snapshot?.companies[id];
              const m = metaFor(id);
              const bk = breakerLabel(c?.breaker_state ?? 0);
              const healthy = c?.healthy ?? false;
              return (
                <div
                  key={id}
                  className="rounded-md border bg-card-2 p-2.5"
                  style={{
                    boxShadow: `inset 2px 0 0 ${m.color}`,
                    borderColor: healthy ? "var(--border)" : "var(--destructive)",
                  }}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-semibold flex items-center gap-1.5 mono">
                      <span
                        className="h-2 w-2 rounded-sm"
                        style={{ background: m.color }}
                      />
                      {m.short.toLowerCase()}
                    </span>
                    <StatusDot up={healthy} />
                  </div>
                  <p className="text-[9.5px] text-muted-foreground mt-0.5">
                    {m.strategy} · {m.multiplier}
                  </p>
                  <div className="mt-2 flex items-center justify-between">
                    <span className="text-[15px] font-semibold tnum mono" style={{ color: m.color }}>
                      {c ? us(c.inference_us.p50) : "—"}
                    </span>
                    <Pill tone={bk.tone}>{bk.label}</Pill>
                  </div>
                  <p className="text-[9px] uppercase tracking-[0.1em] text-label mt-0.5">
                    inference p50 · model v{c ? c.model_version : "—"}
                  </p>
                </div>
              );
            })}
          </div>
        </div>

        <Edge label="outcomes" sub="NotifyOutcome" live={engineUp && aps > 0} reverse />

        {/* Retrainer */}
        <Node
          className="w-[176px] shrink-0"
          icon={<RefreshCw className="h-4 w-4" />}
          title="retrainer"
          tag="Python"
          health={retr?.available ? "up" : "idle"}
          rows={[
            ["status", retr?.available ? retr.status || "idle" : "asleep"],
            ["round", retr?.available ? `#${retr.round}` : "—"],
            ["window", retr?.available ? compact(retr.window) : "—"],
            ["last loss", retr?.available && retr.last_loss ? retr.last_loss.toFixed(3) : "—"],
          ]}
        />
      </div>

      {/* feedback edge: retrainer publishes a new ONNX the fleet hot-swaps */}
      <div className="mt-3 flex items-center gap-3 text-[10px] text-muted-foreground">
        <span className="flex-1 h-px bg-border" />
        <span className="mono flex items-center gap-1.5">
          <RefreshCw className="h-3 w-3" /> retrainer publishes a new ONNX, the fleet hot-swaps it
        </span>
        <span className="flex-1 h-px bg-border" />
      </div>
    </Panel>
  );
}

// A status dot: green/red, with a single restrained blink while up.
function StatusDot({ up }: { up: boolean }) {
  return (
    <span
      className={`h-1.5 w-1.5 rounded-full ${up ? "bg-success animate-live" : "bg-destructive"}`}
    />
  );
}

function Node({
  className,
  icon,
  title,
  tag,
  health,
  rows,
}: {
  className?: string;
  icon: React.ReactNode;
  title: string;
  tag: string;
  health: "up" | "idle" | "down";
  rows: [string, string][];
}) {
  // ring color = health
  const ring = health === "down" ? "var(--destructive)" : "var(--border)";
  return (
    <div
      className={`rounded-md border bg-card-2 p-3 flex flex-col ${className ?? ""}`}
      style={{ borderColor: ring }}
    >
      <div className="flex items-center justify-between">
        <span className="flex items-center gap-1.5 text-[11.5px] font-semibold mono text-foreground">
          {icon}
          {title}
        </span>
        <StatusDot up={health === "up"} />
      </div>
      <span className="text-[9px] mono text-muted-foreground mt-0.5">{tag}</span>
      <div className="mt-2 space-y-1">
        {rows.map(([k, v]) => (
          <div key={k} className="flex items-center justify-between text-[10.5px]">
            <span className="text-label">{k}</span>
            <span className="tnum mono text-foreground truncate ml-2">{v}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// A directed edge between services. The line marches while traffic flows
// (data-driven), and carries a throughput / operation label.
function Edge({
  label,
  sub,
  live,
  reverse,
}: {
  label: string;
  sub: string;
  live?: boolean;
  reverse?: boolean;
}) {
  return (
    <div className="hidden md:flex flex-col items-center justify-center w-16 shrink-0">
      <span className="text-[8.5px] mono text-muted-foreground mb-1">{label}</span>
      <div
        className={`w-full h-px ${live ? `flow-edge ${reverse ? "flow-edge-rev" : ""}` : "bg-border"}`}
      />
      <span className="text-[8px] mono text-label mt-1 truncate max-w-full">{sub}</span>
    </div>
  );
}
