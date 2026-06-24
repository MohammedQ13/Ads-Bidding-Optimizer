"use client";

import { useEffect, useRef, useState } from "react";

// Tweens between values on live updates. Tabular figures.
export function AnimatedNumber({
  value,
  format,
}: {
  value: number;
  format?: (n: number) => string;
}) {
  const [display, setDisplay] = useState(value);
  const fromRef = useRef(value);
  const frameRef = useRef<number | null>(null);

  useEffect(() => {
    const from = fromRef.current;
    const to = value;
    if (from === to) return;
    const duration = 150; // ms
    const start = performance.now();

    function step(now: number) {
      const t = Math.min((now - start) / duration, 1);
      // ease-out
      const eased = 1 - (1 - t) * (1 - t);
      const current = from + (to - from) * eased;
      setDisplay(current);
      fromRef.current = current;
      if (t < 1) {
        frameRef.current = requestAnimationFrame(step);
      } else {
        fromRef.current = to;
        setDisplay(to);
      }
    }

    frameRef.current = requestAnimationFrame(step);
    return () => {
      if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
    };
  }, [value]);

  const text = format ? format(display) : Math.round(display).toLocaleString();
  return <span className="tnum">{text}</span>;
}
