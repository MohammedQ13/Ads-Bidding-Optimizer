"use client";

import { useEffect } from "react";
import { X, RefreshCw, ExternalLink } from "lucide-react";

// About overlay: architecture, benchmarks, and live/replay state.
export function AboutModal({
  open,
  onClose,
  isReplay,
  maxcap = false,
}: {
  open: boolean;
  onClose: () => void;
  isReplay: boolean;
  maxcap?: boolean;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 grid place-items-center p-4 bg-background/80 animate-enter"
      onClick={onClose}
    >
      <div
        className="w-full max-w-2xl max-h-[88vh] overflow-y-auto scroll-thin bg-card border border-border rounded-xl shadow-raised"
        onClick={(e) => e.stopPropagation()}
      >
        {/* header */}
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-hairline sticky top-0 bg-card">
          <div className="flex items-center gap-2.5">
            <span className="grid h-8 w-8 place-items-center rounded-sm border border-border bg-card text-muted-foreground mono text-[11px]">
              RTB
            </span>
            <div>
              <h2 className="text-[14px] font-semibold leading-none">How it works</h2>
            </div>
          </div>
          <button
            onClick={onClose}
            className="h-7 w-7 grid place-items-center rounded-md text-muted-foreground hover:text-foreground hover:bg-muted"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="p-5 space-y-5 text-[12px] leading-relaxed">
          <p className="text-muted-foreground">
            In a first-price auction you pay exactly what you bid, so the
            profit-maximizing bid depends on the whole{" "}
            <span className="text-foreground">distribution of competitors&apos; clearing prices</span>,
            not a single point estimate. The model predicts that distribution; the optimizer
            picks the bid that maximizes{" "}
            <span className="mono text-foreground">E[profit] = (V - b) · P(win at b)</span>.
            Four C++ servers serve those bids over gRPC, a Go exchange runs the auctions, and a
            Python loop retrains the model the fleet hot-swaps while it is serving.
          </p>

          {/* the model */}
          <Section title="The model">
            <p className="text-muted-foreground">
              A discrete-bin network over{" "}
              <span className="text-foreground">301 uniform price bins</span> — bin{" "}
              <span className="mono">k</span> is a clearing price of{" "}
              <span className="mono">k</span> fen — trained on iPinYou season 2 and exported to a
              single self-contained ONNX file. The distribution is what closes the gap: a
              point-estimate regression leaves{" "}
              <span className="mono">~37 fen</span> of regret against the perfect-information oracle,
              while the distributional model reaches{" "}
              <span className="text-foreground mono">~20 fen</span> — about 72% of oracle profit.
            </p>
          </Section>

          {/* architecture */}
          <Section title="Architecture">
            <ul className="space-y-2">
              <Layer n="1" name="Python ML" color="#46b3c2">
                trains the bin distribution and exports ONNX. The optimizer scans the 301-bin
                CDF for the argmax of (V - b)·CDF(b); grid search beats Newton, which diverges
                at high valuations.
              </Layer>
              <Layer n="2" name="C++ inference fleet" color="#7d83e6">
                an async completion-queue gRPC server on ONNX Runtime. A micro-batcher coalesces
                concurrent requests into one inference call, a dedup cache short-circuits repeats,
                and a circuit breaker trips to a heuristic fallback if the model misbehaves. Four
                instances run the same model with different strategies — bid multipliers and a
                budget-paced instance held to a tighter daily budget.
              </Layer>
              <Layer n="3" name="Go auction exchange" color="#a974dd">
                broadcasts each impression to the fleet on a 10ms deadline, draws a synthetic
                field (lognormal bids across eight archetypes), settles the first-price auction,
                paces each strategy&apos;s budget, and aggregates every panel&apos;s telemetry.
              </Layer>
              <Layer n="4" name="Python retrainer" color="#d471b8">
                fine-tunes on a sliding window of recent outcomes, warm-started from the iPinYou
                checkpoint, and writes a new ONNX. The fleet sees the file change, validates the
                graph, and swaps it under a mutex-guarded shared_ptr with no dropped requests.
              </Layer>
            </ul>
          </Section>

          {/* performance */}
          <Section title="Benchmarked performance (c2d-standard-16, 16 vCPU)">
            <p className="text-muted-foreground mb-2.5">
              This system is the <span className="text-foreground">DSP / bidder</span>, not the
              exchange. Its job is to price a bid inside a 10ms deadline, so the headline is
              end-to-end serving latency under load &mdash; not a record throughput.
            </p>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
              <Stat v="~56k" l="bids/sec/node e2e (gRPC) @ p99 2.4ms" />
              <Stat v="~79k" l="req/s peak" />
              <Stat v="~965k" l="bids/sec compute headroom" />
              <Stat v="~1µs" l="per bid (compute)" />
            </div>
            <p className="text-[10.5px] text-muted-foreground mt-2.5 leading-relaxed">
              These are three different measurements. The{" "}
              <span className="text-foreground">compute ceiling</span> is ~965k bids/sec, so the
              model itself is rarely the bottleneck. A node{" "}
              <span className="text-foreground">serves ~56k req/s end-to-end over gRPC at p99
              2.4ms</span>, inside the 10ms deadline (~79k peak). The live market on this dashboard
              runs at ~200/s because the toy exchange is paced and sequential, not because the
              bidder is limited. The node is stateless, so throughput scales with replica count. (A
              budget e2-standard-16 gives ~620k compute on the same code.) Switch the top bar to{" "}
              <span className="text-primary">max</span> for a recorded run with the fleet under
              heavy serving load.
            </p>
          </Section>

          {/* live vs replay */}
          <Section title="Live, replay, max capacity">
            {maxcap ? (
              <div className="space-y-2">
                <p>
                  You&apos;re watching a{" "}
                  <span className="text-primary font-medium">recorded max-capacity run</span>: a
                  load generator saturates the gRPC fleet, so the micro-batcher coalesces
                  requests (batch ~2x) and the tail latency climbs — still inside the 10ms
                  deadline. The demo exchange&apos;s own throughput reads lower here because it
                  shares the busy servers with the load.
                </p>
                <p className="text-muted-foreground">
                  Switch the top bar back to <span className="text-foreground">live</span> for the
                  real-time market (paced ~200/s).
                </p>
              </div>
            ) : isReplay ? (
              <div className="space-y-2">
                <p>
                  You&apos;re watching a{" "}
                  <span className="text-foreground font-medium">recorded demo</span> — a real run
                  that was captured and is being replayed. Every number here was actually measured
                  from the running system (real bids, real gRPC round-trips, real ONNX inference);
                  nothing is simulated. It loops a ~150-frame recording, so all panels work fully.
                </p>
                <p className="text-muted-foreground">
                  The live backend (four C++ servers + the Go exchange + the retrainer) runs on a
                  cloud VM that&apos;s kept off between demos to save credits. When the owner starts
                  it, this console reconnects to real-time data automatically. Switch to{" "}
                  <span className="text-primary">max</span> for a recorded run of the fleet under
                  heavy load.
                </p>
                <button
                  onClick={() => window.location.reload()}
                  className="inline-flex items-center gap-1.5 rounded-md border border-primary/40 bg-primary/10 text-primary px-2.5 py-1 text-[11px] font-medium hover:bg-primary/20"
                >
                  <RefreshCw className="h-3.5 w-3.5" /> Recheck for live backend
                </button>
              </div>
            ) : (
              <p>
                You&apos;re connected to the <span className="text-success font-medium">live system</span> —
                every number is from the running C++ fleet, Go exchange, and retrainer.
              </p>
            )}
          </Section>

          {/* tech + link */}
          <div className="pt-1">
            <p className="mono text-[10px] text-label">
              C++, ONNX Runtime, gRPC, Go, PyTorch, Next.js, Docker, GCP, Vercel
            </p>
          </div>
          <a
            href="https://github.com/MohammedQ13/Ads-Bidding-Optimizer"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 text-[11.5px] text-primary hover:underline"
          >
            <ExternalLink className="h-3.5 w-3.5" /> Source on GitHub
          </a>
        </div>
      </div>
    </div>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <h3 className="label-micro mb-2">{title}</h3>
      {children}
    </div>
  );
}

function Layer({
  n,
  name,
  color,
  children,
}: {
  n: string;
  name: string;
  color: string;
  children: React.ReactNode;
}) {
  return (
    <li className="flex gap-2.5">
      <span className="shrink-0 h-5 w-5 grid place-items-center rounded-sm border border-hairline text-[10px] font-bold mono text-muted-foreground">
        {n}
      </span>
      <span>
        <span className="font-medium" style={{ color }}>{name}</span>
        <span className="text-muted-foreground"> — {children}</span>
      </span>
    </li>
  );
}

function Stat({ v, l }: { v: string; l: string }) {
  return (
    <div className="rounded-md border border-border bg-card-2 p-2">
      <p className="number-display tnum text-[15px] font-semibold text-foreground leading-none">{v}</p>
      <p className="text-[9px] text-muted-foreground mt-1">{l}</p>
    </div>
  );
}
