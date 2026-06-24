import { cn } from "@/lib/utils";
import type { ConnState } from "@/lib/use-scoreboard";

export function LiveDot({ state }: { state: ConnState }) {
  const cfg = {
    live: { dot: "bg-success", text: "text-success", label: "LIVE" },
    connecting: { dot: "bg-warning", text: "text-warning", label: "CONNECTING" },
    offline: { dot: "bg-destructive", text: "text-destructive", label: "OFFLINE" },
  }[state];

  return (
    <span className="inline-flex items-center gap-2">
      <span
        className={cn(
          "h-2 w-2 rounded-full",
          cfg.dot,
          state === "live" && "animate-live"
        )}
      />
      <span
        className={cn(
          "text-[11px] font-medium tracking-[0.1em] tnum",
          cfg.text
        )}
      >
        {cfg.label}
      </span>
    </span>
  );
}
