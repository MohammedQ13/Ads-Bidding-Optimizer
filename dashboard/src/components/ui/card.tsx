import { cn } from "@/lib/utils";

export function Card({
  className,
  hover,
  ...props
}: React.ComponentProps<"div"> & { hover?: boolean }) {
  return (
    <div
      className={cn(
        "bg-card border border-border rounded-lg shadow-card overflow-hidden",
        "transition-all duration-[var(--dur-base)] ease-[var(--ease-out)]",
        hover && "hover:-translate-y-0.5 hover:shadow-raised",
        className
      )}
      {...props}
    />
  );
}
