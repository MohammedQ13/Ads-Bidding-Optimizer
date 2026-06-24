// Records a window of real telemetry from the live engine into a bundled
// replay file, so the dashboard can play back a real run with zero backend.
// Usage: BACKEND=http://34.72.85.177:9200 SECONDS=150 node record-replay.mjs
import { writeFileSync } from "fs";

const BACKEND = process.env.BACKEND || "http://34.72.85.177:9200";
const SECONDS = parseInt(process.env.SECONDS || "150", 10);
const OUT = "src/app/api/stats/replay.json";

const frames = [];
console.log(`recording ${SECONDS}s from ${BACKEND} ...`);
for (let i = 0; i < SECONDS; i++) {
  try {
    const r = await fetch(`${BACKEND}/stats`, { cache: "no-store" });
    if (r.ok) {
      const snap = await r.json();
      if (snap && snap.companies) frames.push(snap);
    }
  } catch (e) {
    console.log("skip frame", i, e.message);
  }
  if (i % 15 === 0) console.log(`  ${i}/${SECONDS}  (${frames.length} frames)`);
  await new Promise((res) => setTimeout(res, 1000));
}

writeFileSync(OUT, JSON.stringify(frames));
console.log(`wrote ${frames.length} frames to ${OUT}`);
