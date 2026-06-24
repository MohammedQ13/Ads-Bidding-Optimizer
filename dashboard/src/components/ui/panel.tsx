import { cn } from "@/lib/utils";

// A panel is the standard section container on the console: a titled card with
// an optional subtitle, leading icon, and a right-hand slot for legends/status.
export function Panel({
  title,
  subtitle,
  icon,
  right,
  info,
  className,
  bodyClassName,
  children,
}: {
  title?: React.ReactNode;
  subtitle?: React.ReactNode;
  icon?: React.ReactNode;
  right?: React.ReactNode;
  info?: string; // one-line "what is this panel" shown behind a ? affordance
  className?: string;
  bodyClassName?: string;
  children: React.ReactNode;
}) {
  return (
    <section
      className={cn(
        "bg-card border border-border rounded-lg shadow-card",
        "animate-enter overflow-hidden",
        className
      )}
    >
      {(title || right) && (
        <header className="flex items-center justify-between gap-3 px-4 py-2.5 border-b border-hairline">
          <div className="flex items-center gap-2 min-w-0">
            {icon && <span className="text-primary shrink-0">{icon}</span>}
            <div className="min-w-0">
              {title && (
                <h2 className="text-[12.5px] font-semibold tracking-tight truncate">
                  {title}
                </h2>
              )}
              {subtitle && (
                <p className="text-[10.5px] text-muted-foreground -mt-0.5 truncate">
                  {subtitle}
                </p>
              )}
            </div>
            {info && <InfoTip text={info} />}
          </div>
          {right && <div className="shrink-0">{right}</div>}
        </header>
      )}
      <div className={cn("p-4", bodyClassName)}>{children}</div>
    </section>
  );
}

// A small "?" affordance that reveals a one-line explanation on hover. Used to
// make jargon-heavy panels self-explanatory without cluttering the header.
export function InfoTip({ text }: { text: string }) {
  return (
    <span className="relative group/info shrink-0 leading-none">
      <span className="grid h-4 w-4 place-items-center rounded-full border border-border text-[9px] text-muted-foreground cursor-help transition-colors hover:text-foreground hover:border-foreground/40">
        ?
      </span>
      <span className="pointer-events-none absolute left-0 top-5 z-40 hidden w-64 rounded-md border border-border bg-card px-2.5 py-1.5 text-[10.5px] font-normal leading-snug tracking-normal text-muted-foreground shadow-raised group-hover/info:block">
        {text}
      </span>
    </span>
  );
}

// A small status badge with a tone color.
export function Pill({
  children,
  tone = "muted",
  className,
}: {
  children: React.ReactNode;
  tone?: "ok" | "warn" | "bad" | "info" | "muted";
  className?: string;
}) {
  const tones: Record<string, string> = {
    ok: "text-success border-success/30 bg-success/10",
    warn: "text-warning border-warning/30 bg-warning/10",
    bad: "text-destructive border-destructive/30 bg-destructive/10",
    info: "text-info border-info/30 bg-info/10",
    muted: "text-muted-foreground border-border bg-muted/40",
  };
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[10px] font-medium tracking-[0.04em] mono",
        tones[tone],
        className
      )}
    >
      {children}
    </span>
  );
}

// A labelled horizontal bar list (used for distributions / breakdowns).
export function Bars({
  items,
  max,
  format,
  className,
}: {
  items: { label: string; value: number; color?: string; sub?: string }[];
  max?: number;
  format?: (n: number) => string;
  className?: string;
}) {
  const top = max ?? Math.max(1, ...items.map((i) => i.value));
  return (
    <div className={cn("space-y-1.5", className)}>
      {items.map((it) => (
        <div key={it.label} className="flex items-center gap-2">
          <span className="w-20 shrink-0 text-[10.5px] text-muted-foreground truncate mono">
            {it.label}
          </span>
          <div className="flex-1 h-3.5 rounded-sm bg-muted/50 overflow-hidden">
            <div
              className="h-full rounded-sm transition-all duration-500 ease-[var(--ease-out)]"
              style={{
                width: `${Math.max(2, (it.value / top) * 100)}%`,
                background: it.color ?? "var(--primary)",
              }}
            />
          </div>
          <span className="w-14 shrink-0 text-right text-[10.5px] tnum mono">
            {format ? format(it.value) : it.value}
          </span>
        </div>
      ))}
    </div>
  );
}
