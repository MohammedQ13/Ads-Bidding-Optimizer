"use client";

import { GitBranch } from "lucide-react";
import { Panel, Pill } from "./ui/panel";
import { metaFor, type AuctionEvent, type Snapshot } from "@/lib/types";
import { us } from "@/lib/utils";

// Distributed-trace waterfall for one auction: the engine broadcasts GetBid to the
// fleet in parallel; each round-trip carries a nested ONNX inference span. We only
// measure two real numbers per bid: the total round-trip (rtt_us) and the inference
// duration (inf_us). We do NOT measure where inside the RPC inference happened, so
// the inference bar is drawn at the tail of the round-trip (server compute just
// before the response returns) — only its width (the measured duration) is exact.

// Pick the newest auction that actually has timed bids to draw as the trace.
function pickTrace(snapshot: Snapshot | null): AuctionEvent | null {
  if (!snapshot?.events?.length) return null;
  for (const ev of snapshot.events) {
    const timed = ev.bids.some((b) => !b.paced && b.rtt_us > 0);
    if (timed) return ev;
  }
  return snapshot.events[0];
}

export function TraceWaterfall({ snapshot }: { snapshot: Snapshot | null }) {
  const ev = pickTrace(snapshot);
  const deadlineUs = (snapshot?.deadline_ms ?? 10) * 1000;

  // order the rows by the engine's company order so colors stay stable
  const order = snapshot?.order?.length
    ? snapshot.order
    : snapshot
    ? Object.keys(snapshot.companies).sort()
    : [];

  const rows = ev
    ? order
        .map((id) => ({ id, bid: ev.bids.find((b) => b.id === id) }))
        .filter((r) => r.bid)
    : [];

  // the trace ends when the slowest bidder returns — that gates the auction
  let resolveUs = 0;
  for (const r of rows) {
    if (r.bid && r.bid.rtt_us > resolveUs) resolveUs = r.bid.rtt_us;
  }
  // leave ~20% headroom so the longest bar doesn't touch the right edge
  const scale = Math.max(resolveUs * 1.2, 1);
  const budgetPct = deadlineUs > 0 ? (resolveUs / deadlineUs) * 100 : 0;

  // a handful of axis ticks in µs
  const ticks: number[] = [];
  for (let i = 0; i <= 4; i++) ticks.push((scale * i) / 4);

  return (
    <Panel
      icon={<GitBranch className="h-4 w-4" />}
      title="Live request trace"
      subtitle="one GetBid auction across the fleet"
      info="One real auction shown like a distributed trace (Jaeger-style): the exchange broadcasts GetBid to all 4 servers at once; each bar is that server's round-trip, with the solid inner bar its ONNX inference time. The auction resolves when the slowest server returns."
      right={
        <Pill tone={ev ? "info" : "muted"}>
          {ev ? `trace ${ev.id}` : "no trace"}
        </Pill>
      }
    >
      {!ev || rows.length === 0 ? (
        <div className="h-40 grid place-items-center text-[12px] text-muted-foreground">
          waiting for a traced auction…
        </div>
      ) : (
        <div>
          {/* trace summary — the headline latency story */}
          <div className="flex flex-wrap items-center gap-x-5 gap-y-1 mb-3 text-[11px]">
            <Meta k="span" v={us(resolveUs)} />
            <Meta k="services" v={`${rows.length + 1}`} />
            <Meta k="fan-out" v={`${rows.length} parallel gRPC`} />
            <Meta
              k="deadline used"
              v={`${budgetPct.toFixed(1)}% of ${snapshot?.deadline_ms ?? 10}ms`}
              tone={budgetPct < 50 ? "ok" : budgetPct < 90 ? "warn" : "bad"}
            />
          </div>

          {/* time axis */}
          <div className="flex">
            <div className="w-[190px] shrink-0" />
            <div className="relative flex-1 h-4 border-b border-hairline">
              {ticks.map((t, i) => (
                <span
                  key={i}
                  className="absolute -top-0.5 text-[9px] mono text-label -translate-x-1/2"
                  style={{ left: `${(t / scale) * 100}%` }}
                >
                  {i === 0 ? "0" : us(t)}
                </span>
              ))}
            </div>
          </div>

          {/* root span: the Go exchange broadcast */}
          <Span
            label="go-exchange"
            op="GetBid · broadcast"
            tag="Go"
            color="var(--muted-foreground)"
            startPct={0}
            widthPct={(resolveUs / scale) * 100}
            timing={us(resolveUs)}
            root
          />

          {/* child spans: one gRPC round-trip per C++ bidder, with the nested
              ONNX inference span placed inside the round-trip window */}
          {rows.map((r) => {
            const b = r.bid!;
            const m = metaFor(r.id);
            const rtt = b.rtt_us;
            const inf = Math.min(b.inf_us, rtt);
            const transport = Math.max(rtt - inf, 0);
            return (
              <Span
                key={r.id}
                label={`cpp-bidder-${m.short.toLowerCase()}`}
                op={`gRPC GetBid${b.fallback ? " · fallback" : ""}`}
                tag={`${m.strategy} ${m.multiplier}`}
                color={m.color}
                startPct={0}
                widthPct={(rtt / scale) * 100}
                timing={us(rtt)}
                fallback={b.fallback}
                // the inference sub-span: width is the measured inf_us; placed at
                // the tail of the round-trip (transport out, then server compute).
                // Only the duration is measured — the placement is a model.
                inf={{
                  startPct: (transport / scale) * 100,
                  widthPct: (inf / scale) * 100,
                  label: us(inf),
                }}
              />
            );
          })}

          {/* the slowest return resolves the auction */}
          <div className="flex mt-1">
            <div className="w-[190px] shrink-0" />
            <div className="relative flex-1 h-5">
              <div
                className="absolute top-0 bottom-0 border-l border-dashed border-border"
                style={{ left: `${(resolveUs / scale) * 100}%` }}
              />
              <span
                className="absolute top-0 text-[9px] mono text-muted-foreground pl-1.5"
                style={{ left: `${(resolveUs / scale) * 100}%` }}
              >
                auction resolves
              </span>
            </div>
          </div>

          <p className="mt-3 text-[10px] text-muted-foreground leading-relaxed">
            <span className="mono text-foreground">{us(resolveUs)}</span> end-to-end
            across {rows.length + 1} services; model time averaged{" "}
            <span className="mono text-foreground">
              {us(rows.reduce((s, r) => s + Math.min(r.bid!.inf_us, r.bid!.rtt_us), 0) / Math.max(rows.length, 1))}
            </span>
            , the rest gRPC transport.
          </p>
        </div>
      )}
    </Panel>
  );
}

function Meta({ k, v, tone }: { k: string; v: string; tone?: "ok" | "warn" | "bad" }) {
  const color =
    tone === "ok"
      ? "text-success"
      : tone === "warn"
      ? "text-warning"
      : tone === "bad"
      ? "text-destructive"
      : "text-foreground";
  return (
    <span className="inline-flex items-baseline gap-1.5">
      <span className="text-[9px] uppercase tracking-[0.1em] text-label">{k}</span>
      <span className={`tnum mono ${color}`}>{v}</span>
    </span>
  );
}

// One span row: a left label column (service · operation, like Jaeger) and a
// timeline bar. Child spans are indented; the root is not.
function Span({
  label,
  op,
  tag,
  color,
  startPct,
  widthPct,
  timing,
  root,
  fallback,
  inf,
}: {
  label: string;
  op: string;
  tag: string;
  color: string;
  startPct: number;
  widthPct: number;
  timing: string;
  root?: boolean;
  fallback?: boolean;
  inf?: { startPct: number; widthPct: number; label: string };
}) {
  return (
    <div className="flex items-center h-7">
      {/* label column */}
      <div className={`w-[190px] shrink-0 min-w-0 ${root ? "" : "pl-3"}`}>
        <div className="flex items-center gap-1.5 min-w-0">
          {!root && (
            <span className="h-2 w-2 rounded-sm shrink-0" style={{ background: color }} />
          )}
          <span className="text-[11px] font-medium truncate mono">{label}</span>
        </div>
        <span className="block text-[9px] text-label truncate ml-3.5">
          {op} · {tag}
        </span>
      </div>

      {/* timeline */}
      <div className="relative flex-1 h-full flex items-center">
        <div
          className="absolute h-3 rounded-[3px]"
          style={{
            left: `${startPct}%`,
            width: `${Math.max(widthPct, 0.6)}%`,
            // root + transport read as a faint fill; the bar is the round-trip
            background: root ? "var(--muted)" : `${color}33`,
            border: fallback ? "1px solid var(--warning)" : "none",
          }}
        />
        {/* nested ONNX inference span — solid, sits inside the round-trip */}
        {inf && (
          <div
            className="absolute h-3 rounded-[3px]"
            style={{
              left: `${inf.startPct}%`,
              width: `${Math.max(inf.widthPct, 0.6)}%`,
              background: color,
            }}
            title={`ONNX inference ${inf.label}`}
          />
        )}
        {/* timing label trailing the bar */}
        <span
          className="absolute text-[9.5px] tnum mono text-muted-foreground whitespace-nowrap"
          style={{ left: `calc(${startPct + Math.max(widthPct, 0.6)}% + 6px)` }}
        >
          {timing}
          {inf && <span className="text-label"> · onnx {inf.label}</span>}
        </span>
      </div>
    </div>
  );
}
