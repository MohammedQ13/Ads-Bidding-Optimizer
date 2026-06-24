// The shape of the telemetry the Go engine serves at /stats. Mirrors the Go
// structs in go-engine/internal/stats. This is one rich snapshot of the whole
// distributed system: per-strategy economics + model output + latency, the
// synthetic market, a live auction feed, the C++ internals (scraped), and the
// retraining loop state.

export interface Latency {
  p50: number;
  p95: number;
  p99: number;
  avg: number;
}

export interface CompanyStats {
  won: number;
  lost: number;
  no_bid: number;
  profit: number;
  spend: number;
  budget: number;
  win_rate: number;
  roi: number;
  bids: number;
  avg_bid: number;
  last_bid: number;
  avg_win_prob: number;
  expected_profit: number;
  fallback: number;
  fallback_rate: number;
  rtt_us: Latency;
  inference_us: Latency;
  breaker_state: number; // 0 closed, 1 open, 2 half-open
  queue_depth: number;
  model_version: number;
  cache_hits: number;
  cold_start: number;
  batch_avg: number;
  served_per_sec: number; // real req/s this server handles (counter-derived)
  healthy: boolean;
}

export interface Bucket {
  lo: number;
  hi: number;
  count: number;
}

export interface Archetype {
  label: string;
  count: number;
  avg_bid: number;
  last_bid: number;
}

export interface EventBid {
  id: string;
  ok: boolean;
  paced: boolean;
  won: boolean;
  bid: number;
  win_prob: number;
  exp_profit: number;
  inf_us: number;
  rtt_us: number;
  fallback: boolean;
}

export interface AuctionEvent {
  id: string;
  t: number;
  hour: number;
  exchange: number;
  slot: string;
  domain: string;
  floor: number;
  clearing: number;
  competitor_top: number;
  winner: string;
  bids: EventBid[];
}

export interface Retrainer {
  available: boolean;
  round: number;
  rows: number;
  last_loss: number;
  exported_at: number;
  model_version: number;
  window: number;
  exports: number;
  status: string;
  loss_history?: number[];
}

export interface Impressions {
  by_hour: number[];
  by_exchange: Record<string, number>;
  by_slot: Record<string, number>;
  floor_rate: number;
}

export interface Snapshot {
  started_at: number;
  uptime_s: number;
  auctions: number;
  auctions_per_sec: number;
  last_clearing_price: number;
  impression_value: number;
  deadline_ms: number;
  num_competitors: number;
  companies: Record<string, CompanyStats>;
  order: string[];
  histograms: Record<string, Bucket[]>;
  win_prob_hist: Bucket[];
  impressions: Impressions;
  competitor_archetypes: Archetype[];
  events: AuctionEvent[];
  retrainer: Retrainer;
  // set by the API route when serving a recorded run (backend offline)
  _replay?: boolean;
  // set when serving the "max capacity" recorded run (the fleet under load)
  _maxcap?: boolean;
}

// Display metadata for each strategy. color is a hex so it works inside
// SVG/recharts stroke attributes. Each strategy is treated like a distinct
// SERVICE (the way Tempo/Grafana color one service per trace): muted, mid-
// saturation hues from the Grafana classic series palette — distinguishable
// but never neon, and kept clear of the green/amber/red reserved for status.
export interface CompanyMeta {
  name: string;
  short: string;
  strategy: string;
  color: string;
  multiplier: string;
}

// one cohesive series palette: a teal -> indigo -> violet -> magenta sweep, all at
// similar lightness/saturation so the four read as one family on the charcoal canvas.
// kept in sync with the --company-* tokens in globals.css.
export const COMPANY_META: Record<string, CompanyMeta> = {
  "company-a": { name: "Company A", short: "A", strategy: "Profit-Max", color: "#46b3c2", multiplier: "1.0x" },
  "company-b": { name: "Company B", short: "B", strategy: "Aggressive", color: "#7d83e6", multiplier: "1.2x" },
  "company-c": { name: "Company C", short: "C", strategy: "Conservative", color: "#a974dd", multiplier: "0.8x" },
  "company-d": { name: "Company D", short: "D", strategy: "Budget-Paced", color: "#d471b8", multiplier: "1.0x" },
};

const FALLBACK_COLORS = ["#7d83e6", "#46b3c2", "#a974dd", "#d471b8", "#6e7bb0"];

export function metaFor(id: string): CompanyMeta {
  if (COMPANY_META[id]) return COMPANY_META[id];
  // stable-ish color for unknown ids
  let h = 0;
  for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) >>> 0;
  return {
    name: id,
    short: id.slice(-1).toUpperCase(),
    strategy: "Strategy",
    color: FALLBACK_COLORS[h % FALLBACK_COLORS.length],
    multiplier: "",
  };
}

export function breakerLabel(code: number): { label: string; tone: "ok" | "warn" | "bad" } {
  if (code === 1) return { label: "OPEN", tone: "bad" };
  if (code === 2) return { label: "HALF", tone: "warn" };
  return { label: "CLOSED", tone: "ok" };
}
