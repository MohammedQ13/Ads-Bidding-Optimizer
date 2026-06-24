// Server-side proxy to the engine's /stats, with an automatic replay fallback.
//
// Live path: the browser calls this same-origin route; the Vercel server (not the
// browser) fetches the HTTP backend — so the frontend can be HTTPS-on-Vercel
// while the backend is plain HTTP, with no mixed-content block and no CORS.
//
// Replay path: if the backend is unreachable (VM stopped / credits exhausted),
// we serve a bundled recording of a real run, looped and advanced by wall-clock,
// so the showcase is always live-looking at zero compute cost. Replayed snapshots
// carry `_replay: true` so the UI can label them honestly.
//
// BACKEND_URL (server env, NOT NEXT_PUBLIC) overrides the live target. Locally we
// use 127.0.0.1 (IPv4 on purpose — Node's fetch can pick IPv6 ::1 and miss the
// Docker port mapping). On Vercel we default to the GCP backend's static IP.
import replay from "./replay.json";
import replayMaxcap from "./replay_maxcap.json";

export const dynamic = "force-dynamic";
export const revalidate = 0;

// loop a recorded run, advanced ~1 frame/sec by wall-clock
function serveReplay(
  frames: Record<string, unknown>[],
  extra: Record<string, unknown>,
) {
  const idx = Math.floor(Date.now() / 1000) % frames.length;
  const snap = { ...frames[idx], ...extra };
  return new Response(JSON.stringify(snap), {
    status: 200,
    headers: { ...NO_STORE, "x-source": "replay" },
  });
}

const BACKEND =
  process.env.BACKEND_URL ||
  (process.env.VERCEL ? "http://34.72.85.177:9200" : "http://127.0.0.1:9200");

const NO_STORE = {
  "content-type": "application/json",
  "cache-control": "no-store",
};

export async function GET(request: Request) {
  // "Max capacity" mode: always serve the recorded full-throttle run (the fleet
  // under real bench load), regardless of whether the live backend is up - the
  // live market is paced, so this recorded run is the only way to show the fleet
  // working hard (micro-batcher batching, real under-load p99).
  const mode = new URL(request.url).searchParams.get("mode");
  if (mode === "maxcap") {
    const frames = replayMaxcap as unknown as Record<string, unknown>[];
    if (Array.isArray(frames) && frames.length > 0) {
      return serveReplay(frames, { _replay: true, _maxcap: true });
    }
  }

  // 1) try the live backend (short timeout so we fall back fast)
  try {
    const r = await fetch(`${BACKEND}/stats`, {
      cache: "no-store",
      signal: AbortSignal.timeout(1500),
    });
    if (r.ok) {
      const body = await r.text();
      return new Response(body, {
        status: 200,
        headers: { ...NO_STORE, "x-source": "live" },
      });
    }
  } catch {
    // fall through to replay
  }

  // 2) replay fallback: loop the recorded run, advanced ~1 frame/sec
  const frames = replay as unknown as Record<string, unknown>[];
  if (Array.isArray(frames) && frames.length > 0) {
    return serveReplay(frames, { _replay: true });
  }

  // 3) nothing to show
  return Response.json({ error: "offline" }, { status: 503 });
}
