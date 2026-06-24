import { Crown } from "lucide-react";
import { Card } from "./ui/card";
import { Pill } from "./ui/panel";
import { Sparkline } from "./sparkline";
import { AnimatedNumber } from "./animated-number";
import { breakerLabel, metaFor, type CompanyStats } from "@/lib/types";
import { fmt, pct, us } from "@/lib/utils";

export function CompanyCard({
  id,
  stats,
  series,
  dailyBudget,
  leader,
}: {
  id: string;
  stats: CompanyStats;
  series: number[];
  dailyBudget: number;
  leader?: boolean;
}) {
  const m = metaFor(id);
  const budgetPct =
    dailyBudget > 0 ? Math.max(0, Math.min(1, stats.budget / dailyBudget)) : 0;
  const bk = breakerLabel(stats.breaker_state);

  return (
    <Card hover className="p-3.5 flex flex-col gap-2.5">
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-2">
          <span
            className="h-2.5 w-2.5 rounded-full shrink-0"
            style={{ background: m.color }}
          />
          <div>
            <p className="text-[12.5px] font-semibold leading-none">{m.name}</p>
            <p className="text-[10px] text-muted-foreground mt-1 mono">
              {m.strategy} · {m.multiplier}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-1.5">
          {leader && (
            <span className="inline-flex items-center gap-1 text-[9.5px] font-medium text-warning">
              <Crown className="h-3.5 w-3.5" /> LEAD
            </span>
          )}
          <span
            className={`h-1.5 w-1.5 rounded-full ${
              stats.healthy ? "bg-success animate-live" : "bg-destructive"
            }`}
          />
        </div>
      </div>

      <div>
        <p className="text-[9px] uppercase tracking-[0.08em] text-label">Profit</p>
        <p
          className="tnum mono text-[23px] font-semibold tracking-[-0.015em] leading-none mt-1"
          style={{ color: stats.profit < 0 ? "var(--destructive)" : m.color }}
        >
          <AnimatedNumber value={stats.profit} format={fmt} />
        </p>
      </div>

      <Sparkline data={series} color={m.color} className="h-7 w-full" />

      <div className="grid grid-cols-3 gap-x-2 gap-y-2 pt-0.5">
        <Metric label="Win rate" value={pct(stats.win_rate)} />
        <Metric label="Avg bid" value={fmt(stats.avg_bid)} />
        <Metric label="P(win)" value={pct(stats.avg_win_prob)} />
        <Metric label="RTT p99" value={us(stats.rtt_us.p99)} />
        <Metric label="Inf p50" value={us(stats.inference_us.p50)} />
        <Metric label="Fallback" value={pct(stats.fallback_rate)} />
      </div>

      <div className="flex items-center justify-between pt-0.5">
        <Pill tone={bk.tone}>BRK {bk.label}</Pill>
        <span className="text-[9.5px] mono text-muted-foreground">
          q{stats.queue_depth} · v{stats.model_version}
        </span>
      </div>

      <div>
        <div className="flex items-center justify-between text-[9px] uppercase tracking-[0.06em] text-label mb-1">
          <span>Budget</span>
          <span className="tnum mono normal-case">{fmt(stats.budget)}</span>
        </div>
        <div className="h-1.5 rounded-full bg-muted overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-500 ease-[var(--ease-out)]"
            style={{ width: `${budgetPct * 100}%`, background: m.color }}
          />
        </div>
      </div>
    </Card>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-[9px] uppercase tracking-[0.05em] text-label">{label}</p>
      <p className="tnum mono text-[12px] font-medium mt-0.5 leading-none">{value}</p>
    </div>
  );
}
