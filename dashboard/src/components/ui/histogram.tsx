"use client";

import type { Bucket } from "@/lib/types";

// A compact vertical-bar histogram drawn as inline SVG. Used for price/bid/
// win-probability distributions. Optionally overlays a second series for
// side-by-side comparison (e.g. clearing price vs our bids).
export function Histogram({
  buckets,
  color = "var(--primary)",
  overlay,
  overlayColor = "var(--muted-foreground)",
  height = 90,
  unit = "",
  marker,
  markerColor = "var(--warning)",
  normalizeSeparately = false,
  openTop = true,
}: {
  buckets: Bucket[];
  color?: string;
  overlay?: Bucket[];
  overlayColor?: string;
  height?: number;
  unit?: string;
  marker?: number; // a vertical reference line at this x value
  markerColor?: string;
  // price histograms have an open top bucket (200+); a bounded series (e.g. win
  // probability 95-100) should pass openTop={false} so the last label isn't "95+"
  openTop?: boolean;
  // when overlaying two series with very different totals, scale each to its own
  // peak so the shapes are comparable instead of the taller series flattening the
  // other. Counts are still shown on hover, so the absolute scale isn't lost.
  normalizeSeparately?: boolean;
}) {
  if (!buckets || buckets.length === 0) {
    return (
      <div className="grid place-items-center text-[10.5px] text-muted-foreground" style={{ height }}>
        no data yet
      </div>
    );
  }
  let max = 1;
  let overlayMax = 1;
  for (let i = 0; i < buckets.length; i++) {
    if (buckets[i].count > max) max = buckets[i].count;
    if (overlay && overlay[i] && overlay[i].count > overlayMax) overlayMax = overlay[i].count;
  }
  // shared scale unless asked to normalize each series to its own peak
  if (!normalizeSeparately && overlayMax > max) max = overlayMax;
  const oMax = normalizeSeparately ? overlayMax : max;
  const n = buckets.length;
  const gap = 2;
  const barH = height - 16;

  return (
    <div>
      <div className="flex items-end gap-[2px]" style={{ height: barH }}>
        {buckets.map((b, i) => {
          const h = (b.count / max) * barH;
          const oh = overlay && overlay[i] ? (overlay[i].count / oMax) * barH : 0;
          const isMarker = marker != null && marker >= b.lo && marker < b.hi;
          return (
            <div
              key={i}
              className="flex-1 relative group flex items-end justify-center"
              style={{ height: barH }}
            >
              {isMarker && (
                <span
                  className="absolute inset-y-0 left-1/2 w-px -translate-x-1/2"
                  style={{ background: markerColor }}
                />
              )}
              <div className="w-full flex items-end justify-center gap-[1px]">
                <div
                  className="rounded-t-[2px] transition-all duration-500"
                  style={{
                    height: Math.max(b.count > 0 ? 1.5 : 0, h),
                    width: overlay ? "50%" : "100%",
                    background: color,
                  }}
                />
                {overlay && (
                  <div
                    className="rounded-t-[2px] transition-all duration-500"
                    style={{
                      height: Math.max(overlay[i]?.count > 0 ? 1.5 : 0, oh),
                      width: "50%",
                      background: overlayColor,
                    }}
                  />
                )}
              </div>
              <span className="pointer-events-none absolute -top-5 left-1/2 -translate-x-1/2 whitespace-nowrap rounded bg-card border border-border px-1 py-0.5 text-[9px] mono opacity-0 group-hover:opacity-100 z-10">
                {b.lo}{unit} · {b.count}
              </span>
            </div>
          );
        })}
      </div>
      <div className="flex justify-between mt-1.5 text-[8.5px] mono text-label" style={{ gap }}>
        <span>{buckets[0].lo}{unit}</span>
        <span>{buckets[Math.floor(n / 2)].lo}{unit}</span>
        <span>{buckets[n - 1].lo}{openTop ? "+" : ""}{unit}</span>
      </div>
    </div>
  );
}
