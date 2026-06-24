"use client";

import { RefreshCw } from "lucide-react";
import { Panel, Pill } from "./ui/panel";
import { Sparkline } from "./sparkline";
import type { Snapshot } from "@/lib/types";
import { ago, compact } from "@/lib/utils";

// The feedback loop: the Python retrainer reads a sliding window of auction
// outcomes, fine-tunes the bins model warm-started from the iPinYou checkpoint,
// and publishes a new ONNX that the C++ fleet hot-swaps.
export function RetrainerPanel({ snapshot }: { snapshot: Snapshot | null }) {
  const r = snapshot?.retrainer;
  const nowMs = Date.now();
  // on recorded data (replay / max-capacity) nothing is actually hot-swapping,
  // so don't show the live "swapping now" pulse — it would imply activity that
  // isn't happening. Only the live backend gets the animated indicator.
  const recorded = !!snapshot?._replay;

  const statusTone =
    r?.status === "exported"
      ? "ok"
      : r?.status === "training"
      ? "info"
      : r?.status === "waiting"
      ? "warn"
      : "muted";

  return (
    <Panel
      icon={<RefreshCw className="h-4 w-4" />}
      title="Retraining loop"
      subtitle="sliding-window fine-tune, hot-reload on export"
      info="The feedback loop: the Python retrainer fine-tunes the model on a sliding window of recent auction outcomes, exports a new ONNX, and the C++ fleet hot-swaps it under a lock with no dropped requests. Loss stays flat because it is a fine-tune around an already-converged warm start."
      right={
        <Pill tone={r?.available ? (statusTone as "ok") : "bad"}>
          {r?.available ? (r.status || "idle").toUpperCase() : "ASLEEP"}
        </Pill>
      }
    >
      {!r?.available ? (
        <div className="h-28 grid place-items-center text-center text-[11px] text-muted-foreground px-6">
          retrainer hasn&apos;t published a status yet — it warms up once enough
          auction outcomes have accumulated in the shared window.
        </div>
      ) : (
        <div className="space-y-4">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <Stat label="Round" value={`#${r.round}`} />
            <Stat label="Window rows" value={compact(r.rows)} sub={`cap ${compact(r.window)}`} />
            <Stat label="Last loss" value={r.last_loss ? r.last_loss.toFixed(4) : "—"} />
            <Stat label="Exports" value={`${r.exports}`} sub={`exported v${r.model_version}`} />
          </div>

          <div>
            <div className="flex items-center justify-between mb-1">
              <p className="text-[10px] uppercase tracking-[0.08em] text-label">
                Cross-entropy loss <span className="normal-case tracking-normal text-muted-foreground/70">(fine-tune around a converged warm-start, so it stays flat)</span>
              </p>
              <p className="text-[9.5px] mono text-muted-foreground">
                last export {ago(r.exported_at, nowMs)}
              </p>
            </div>
            {r.loss_history && r.loss_history.length > 1 ? (
              <Sparkline data={r.loss_history} color="#3d71d9" className="h-12 w-full" />
            ) : (
              <div className="h-12 grid place-items-center text-[10px] text-muted-foreground">
                building loss curve…
              </div>
            )}
          </div>

          <div className="flex items-center gap-2 text-[10px] text-muted-foreground border-t border-hairline pt-2.5">
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                recorded ? "bg-muted-foreground/50" : "bg-success animate-live"
              }`}
            />
            <span className="mono">
              {recorded
                ? "hot-reload path: mtime watch, validate, atomic shared_ptr swap"
                : "watching for new model: mtime watch, validate, atomic shared_ptr swap"}
            </span>
          </div>
        </div>
      )}
    </Panel>
  );
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-md border border-border bg-card-2 p-2.5">
      <p className="text-[9px] uppercase tracking-[0.08em] text-label">{label}</p>
      <p className="text-[17px] font-semibold tnum mono mt-1 leading-none">{value}</p>
      {sub && <p className="text-[9px] text-muted-foreground mt-1 mono">{sub}</p>}
    </div>
  );
}
