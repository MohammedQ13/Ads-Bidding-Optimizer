"use client";

import { Radio } from "lucide-react";
import { Panel, Pill } from "./ui/panel";
import { metaFor, type AuctionEvent, type Snapshot } from "@/lib/types";
import { us } from "@/lib/utils";

// Scrolling feed of recent auctions. Each row shows every strategy's bid (with
// model win-prob and server inference time), the field's top bid, the clearing
// price, and the winner.
export function AuctionStream({ snapshot }: { snapshot: Snapshot | null }) {
  const events = (snapshot?.events ?? []).slice(0, 16);
  const recorded = !!snapshot?._replay || !!snapshot?._maxcap;
  const ids = snapshot?.order?.length
    ? snapshot.order
    : snapshot
    ? Object.keys(snapshot.companies).sort()
    : [];

  return (
    <Panel
      icon={<Radio className="h-4 w-4" />}
      title="Live auction stream"
      subtitle="newest first"
      right={
        <Pill tone={events.length ? "ok" : "muted"}>
          {events.length ? (recorded ? "replay" : "live") : "idle"}
        </Pill>
      }
      bodyClassName="p-0"
    >
      <div className="max-h-[420px] overflow-y-auto scroll-thin">
        {events.length === 0 && (
          <div className="h-32 grid place-items-center text-[11px] text-muted-foreground">
            waiting for auctions…
          </div>
        )}
        {events.map((ev) => (
          <Row key={ev.id} ev={ev} ids={ids} />
        ))}
      </div>
    </Panel>
  );
}

function Row({ ev, ids }: { ev: AuctionEvent; ids: string[] }) {
  const bidById: Record<string, AuctionEvent["bids"][number]> = {};
  for (const b of ev.bids) bidById[b.id] = b;

  const winnerIsCompany = ev.winner && ev.winner !== "market" && ev.winner !== "backtest";
  const winMeta = winnerIsCompany ? metaFor(ev.winner) : null;

  return (
    <div className="animate-rowflash flex items-center gap-3 px-4 py-2 border-b border-hairline/60 text-[11px]">
      {/* request meta */}
      <div className="w-[120px] shrink-0 min-w-0">
        <p className="mono font-medium truncate">{ev.id}</p>
        <p className="text-[9.5px] text-muted-foreground mono truncate">
          {ev.slot} · {String(ev.hour).padStart(2, "0")}:00 · ADX{ev.exchange}
        </p>
      </div>

      {/* per-company bid chips */}
      <div className="flex-1 flex flex-wrap gap-1.5 min-w-0">
        {ids.map((id) => {
          const b = bidById[id];
          const m = metaFor(id);
          if (!b || b.paced || !b.ok) {
            return (
              <span
                key={id}
                className="inline-flex items-center gap-1 rounded border border-border/60 px-1.5 py-0.5 text-[9.5px] mono text-muted-foreground/60"
                title={b?.paced ? "paced out (budget)" : "no bid"}
              >
                {m.short} {b?.paced ? "paced" : "—"}
              </span>
            );
          }
          return (
            <span
              key={id}
              className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[9.5px] mono"
              style={{
                border: `1px solid ${m.color}${b.won ? "" : "44"}`,
                background: b.won ? `${m.color}22` : "transparent",
                color: m.color,
              }}
              title={`P(win) ${Math.round(b.win_prob * 100)}% · inf ${us(b.inf_us)}${b.fallback ? " · fallback" : ""}`}
            >
              {m.short} {Math.round(b.bid)}
              {b.fallback && <span className="text-warning">!</span>}
              <span className="opacity-50">{Math.round(b.win_prob * 100)}%</span>
            </span>
          );
        })}
      </div>

      {/* market + outcome */}
      <div className="w-[150px] shrink-0 flex items-center justify-end gap-2 mono">
        <span className="text-[9.5px] text-muted-foreground" title="synthetic field top bid">
          fld {Math.round(ev.competitor_top)}
        </span>
        <span className="text-[10.5px] tnum" title="clearing price">
          clr {Math.round(ev.clearing)}
        </span>
        {winnerIsCompany ? (
          <span
            className="inline-flex items-center rounded px-1.5 py-0.5 text-[9.5px] font-semibold"
            style={{ background: `${winMeta!.color}22`, color: winMeta!.color, border: `1px solid ${winMeta!.color}55` }}
          >
            {winMeta!.short} WON
          </span>
        ) : (
          <span className="inline-flex items-center rounded border border-border px-1.5 py-0.5 text-[9.5px] text-muted-foreground">
            FIELD
          </span>
        )}
      </div>
    </div>
  );
}
