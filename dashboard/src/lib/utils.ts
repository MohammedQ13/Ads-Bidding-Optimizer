import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// fen -> a short money-ish string (the dataset is in fen; we show raw fen)
export function fmt(n: number): string {
  if (!isFinite(n)) return "0";
  const abs = Math.abs(n);
  if (abs >= 1_000_000) return (n / 1_000_000).toFixed(2) + "M";
  if (abs >= 1_000) return (n / 1_000).toFixed(1) + "k";
  return Math.round(n).toLocaleString();
}

export function pct(n: number): string {
  return (n * 100).toFixed(1) + "%";
}

// microseconds -> a compact latency string (µs under 1ms, ms above)
export function us(n: number): string {
  if (!isFinite(n) || n <= 0) return "—";
  if (n >= 1000) return (n / 1000).toFixed(n >= 10000 ? 0 : 1) + "ms";
  return Math.round(n) + "µs";
}

export function compact(n: number): string {
  if (!isFinite(n)) return "0";
  const abs = Math.abs(n);
  if (abs >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (abs >= 1_000) return (n / 1_000).toFixed(1) + "k";
  return Math.round(n).toString();
}

// seconds -> "1h 23m" style uptime
export function uptime(s: number): string {
  if (!isFinite(s) || s <= 0) return "0s";
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${sec}s`;
  return `${sec}s`;
}

// unix seconds -> "12s ago" style relative time
export function ago(unixSec: number, nowMs: number): string {
  if (!unixSec) return "never";
  const d = Math.max(0, Math.floor(nowMs / 1000 - unixSec));
  if (d < 60) return `${d}s ago`;
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  return `${Math.floor(d / 3600)}h ago`;
}
