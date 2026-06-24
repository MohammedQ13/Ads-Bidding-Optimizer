"use client";

import { useState } from "react";
import { Activity, Cpu, HelpCircle } from "lucide-react";
import { LiveDot } from "./live-dot";
import { ThemeToggle } from "./theme-toggle";
import { AboutModal } from "./about-modal";
import type { ConnState } from "@/lib/use-scoreboard";
import type { Snapshot } from "@/lib/types";
import { compact, uptime } from "@/lib/utils";

// The top command bar: system identity, connection state, and the few numbers
// you always want in view (throughput, total auctions, uptime).
export function CommandBar({
  snapshot,
  state,
  mode,
  setMode,
}: {
  snapshot: Snapshot | null;
  state: ConnState;
  mode: "live" | "maxcap";
  setMode: (m: "live" | "maxcap") => void;
}) {
  const [aboutOpen, setAboutOpen] = useState(false);
  const isMaxcap = !!snapshot?._maxcap;
  const isReplay = !!snapshot?._replay && !isMaxcap;
  // real serving throughput across the whole fleet (counter-derived req/s), the
  // honest "throughput" headline - the exchange's own auction rate stays ~100-200/s
  let fleetServed = 0;
  if (snapshot) {
    for (const id of Object.keys(snapshot.companies)) {
      fleetServed += snapshot.companies[id].served_per_sec ?? 0;
    }
  }
  return (
    <header className="sticky top-0 z-30 border-b border-border bg-background">
      <div className="mx-auto max-w-[1500px] px-4 lg:px-6 h-14 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3 min-w-0">
          <span className="grid h-9 w-9 place-items-center rounded-sm border border-border bg-card text-muted-foreground">
            <Cpu className="h-4.5 w-4.5" />
          </span>
          <div className="leading-tight min-w-0">
            <p className="text-[13.5px] font-semibold tracking-tight">
              RTB Control Plane
            </p>
            <p className="text-[10.5px] text-muted-foreground -mt-0.5 truncate">
              ML price model → C++ gRPC fleet → Go auctions → live retraining
            </p>
          </div>
        </div>

        <div className="flex items-center gap-4 lg:gap-6">
          <BarStat
            icon={<Activity className="h-3.5 w-3.5" />}
            label="fleet req/s"
            value={snapshot ? `${compact(fleetServed)}/s` : "—"}
            accent
          />
          <BarStat
            label="exchange"
            value={snapshot ? `${compact(snapshot.auctions_per_sec)}/s auc` : "—"}
          />
          <BarStat
            label="uptime"
            value={snapshot ? uptime(snapshot.uptime_s) : "—"}
          />
          <ModeToggle mode={mode} setMode={setMode} />
          {isMaxcap ? (
            <button
              onClick={() => setAboutOpen(true)}
              className="inline-flex items-center gap-1.5 rounded-md border border-primary/40 bg-primary/10 px-2 py-0.5 hover:bg-primary/20 transition-colors"
              title="recorded run with the gRPC fleet under heavy concurrent load: the micro-batcher coalesces requests (batch ~2x) and tail latency climbs, still inside the 10ms deadline"
            >
              <span className="h-2 w-2 rounded-full bg-primary animate-live" />
              <span className="text-[11px] font-medium tracking-[0.1em] text-primary mono">
                MAX CAPACITY
              </span>
            </button>
          ) : isReplay ? (
            <button
              onClick={() => setAboutOpen(true)}
              className="inline-flex items-center gap-1.5 rounded-md border border-border bg-muted/40 px-2 py-0.5 text-muted-foreground transition-colors hover:text-foreground hover:border-foreground/40"
              title="Recorded demo — a real captured run replayed because the live backend VM is currently off. Click for details."
            >
              <span className="h-2 w-2 rounded-full bg-muted-foreground/60" />
              <span className="text-[11px] font-medium tracking-[0.1em] mono">
                RECORDED DEMO
              </span>
              <HelpCircle className="h-3 w-3 opacity-70" />
            </button>
          ) : (
            <LiveDot state={state} />
          )}
          <button
            aria-label="How it works"
            onClick={() => setAboutOpen(true)}
            className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-border bg-card text-muted-foreground transition-colors hover:text-foreground hover:bg-muted"
          >
            <HelpCircle className="h-4 w-4" />
          </button>
          <ThemeToggle />
        </div>
      </div>
      <AboutModal
        open={aboutOpen}
        onClose={() => setAboutOpen(false)}
        isReplay={isReplay}
        maxcap={isMaxcap}
      />
    </header>
  );
}

function BarStat({
  icon,
  label,
  value,
  accent,
}: {
  icon?: React.ReactNode;
  label: string;
  value: string;
  accent?: boolean;
}) {
  return (
    <div className="hidden sm:flex flex-col items-end leading-none">
      <span
        className={`text-[14px] font-semibold tnum mono flex items-center gap-1 ${
          accent ? "text-primary" : "text-foreground"
        }`}
      >
        {icon}
        {value}
      </span>
      <span className="text-[9px] uppercase tracking-[0.14em] text-label mt-1">
        {label}
      </span>
    </div>
  );
}

// Live vs. Max-capacity toggle. "Live" is the paced market (or replay fallback);
// "Max" serves a recorded run with a load generator saturating the gRPC fleet, so
// the micro-batcher coalesces requests (batch ~2x) and tail latency climbs (still
// under the 10ms deadline). The demo exchange's own throughput drops because it
// shares the servers with the load — so aps reads lower, not higher.
function ModeToggle({
  mode,
  setMode,
}: {
  mode: "live" | "maxcap";
  setMode: (m: "live" | "maxcap") => void;
}) {
  return (
    <div className="hidden md:inline-flex items-center rounded-md border border-border bg-card p-0.5 mono text-[10.5px]">
      <button
        onClick={() => setMode("live")}
        className={`px-2 py-0.5 rounded-[4px] transition-colors ${
          mode === "live"
            ? "bg-muted text-foreground"
            : "text-muted-foreground hover:text-foreground"
        }`}
        title="the live market (paced ~200/s) or recorded replay if the backend is off"
      >
        live
      </button>
      <button
        onClick={() => setMode("maxcap")}
        className={`px-2 py-0.5 rounded-[4px] transition-colors ${
          mode === "maxcap"
            ? "bg-primary/15 text-primary"
            : "text-muted-foreground hover:text-foreground"
        }`}
        title="recorded run with the gRPC fleet under heavy concurrent load (micro-batcher engaging, elevated tail latency)"
      >
        max
      </button>
    </div>
  );
}
