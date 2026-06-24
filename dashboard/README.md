# RTB Control Plane: live telemetry console

A real-time observability console for the whole real-time-bidding system: an ML
clearing-price model, four low-latency C++ inference servers that share one ONNX
model, a Go auction exchange, and a Python sliding-window retraining loop. One
page shows the system end to end.

Built with Next.js 16 + React 19 + Tailwind v4 + Recharts. The look is a calm
Grafana-style charcoal console (single blue accent, monospace numbers, hairline
borders, status colors only where they mean something). A light theme ships too.

Live: https://rtb-control-plane.vercel.app

## What the project does (30-second version)

In a first-price auction you pay what you bid, so the profit-maximizing bid needs
the whole distribution of competitors' clearing prices, not a single guess. The
model predicts that distribution over 301 price bins and the optimizer picks the
bid that maximizes `E[profit] = (V - b) * P(win at b)`. A point estimate leaves
about 37 fen of regret against the perfect-information oracle; the distributional
model gets to about 20 fen, roughly 72% of oracle profit. Trained on iPinYou
season 2.

The four servers run the same model with different strategies:

- Company A: Profit-Max, 1.0x bid
- Company B: Aggressive, 1.2x bid (overbids, so it can book a real loss)
- Company C: Conservative, 0.8x bid (wins less, higher margin)
- Company D: Budget-Paced, 1.0x bid but a smaller daily budget, so it spends its
  share early and sits out the rest of the day (wins about half of A)

## Three view modes

- **Live**: real-time data from the running backend (when the VM is on). Numbers
  tick as auctions happen.
- **Recorded demo**: when the backend VM is off, the page replays a real captured
  run (a ~150-frame recording). Every number was actually measured; nothing is
  faked. So the public link is never dark and costs nothing.
- **Max**: a recorded run with the gRPC fleet under heavy concurrent load (a load
  generator saturates the servers). The micro-batcher engages (batch size climbs
  from ~1 to ~8), tail latency rises but stays under the 10ms deadline.

## Two throughput numbers (they are different on purpose)

- **Exchange rate** (~100-200/s): how fast the small Go auctioneer runs auctions.
  It is sequential and paced, so it is deliberately modest.
- **Fleet req/s**: the real number of bid requests the four C++ servers handle,
  read from each server's `bid_request_total` counter (rate from the delta between
  scrapes). About 500/s live; about 30k/s in the max-capacity run.

The gray strip at the top is separate: benchmarked ceilings measured offline on a
16-vCPU box (~56k req/s end to end at p99 2.4ms, ~79k peak, ~965k compute). The
demo VM is smaller, so its honest under-load number is ~30k.

## Panels

- **Service map**: the live wiring: the Go exchange fanning `GetBid` to the C++
  fleet, outcomes streaming to the retrainer, the retrainer publishing a model the
  fleet hot-swaps. Per-node health.
- **KPI strip**: exchange rate, fleet p99 round-trip, ONNX inference p50, clearing
  price, fallback rate, total profit, fleet req/s, model version, each with a trend.
- **Live request trace**: one auction as a distributed trace: the broadcast to all
  four servers in parallel, each round-trip with its ONNX inference inside.
- **Latency / RED**: rate, fallback, and round-trip p50/p95/p99 per server against
  the 10ms deadline, plus the inference-time distribution.
- **Latency heatmap**: round-trip latency over time, by band, with p50/p99 lines.
- **Cumulative profit**: the four strategies over time (B can go negative).
- **Strategy cards**: per strategy: profit, win rate, avg bid, predicted P(win),
  latency, fallback, breaker state, queue depth, model version, budget.
- **Live auction stream**: recent auctions: each strategy's bid, win-prob, paced
  flags, the field's top bid, clearing price, and who won.
- **Model predictions**: bid -> P(win) cloud, win-prob distribution, calibration
  (predicted vs realized win rate), expected vs booked profit.
- **Market & impressions**: competitor archetypes by average bid, price
  distributions, and the impression mix by hour, exchange, and slot size.
- **Micro-batcher saturation**: queue depth and average batch size per server
  (~1 at rest, ~8 under load).
- **C++ serving internals**: inference p99, batcher queue/size, dedup cache, cold
  starts, model version, breaker, scraped from each server's `/metrics`.
- **Retraining loop**: round, window size, loss, exports, model version, and the
  zero-downtime hot-reload status.

Most panels have a small `?` that explains them in one line. The "How it works"
button (top right) opens a fuller writeup.

## How it connects

The Go engine is the single telemetry aggregator:

- `GET /stats`: one JSON snapshot of the whole system (schema in `src/lib/types.ts`)
- `GET /events`: SSE stream pushing the snapshot ~twice a second

The browser only calls the same-origin `/api/stats` proxy, which fetches the HTTP
backend server-side. So the page can be HTTPS on Vercel while the backend is plain
HTTP, with no mixed-content block and no CORS. Locally the proxy targets
`127.0.0.1:9200` (IPv4 on purpose); on Vercel it targets the GCP static IP. If the
backend is unreachable the proxy serves the bundled recording instead.

## Run locally

```
npm install
npm run dev                     # http://localhost:3000
```

For live data, start the backend from the repo root:
```
docker compose up -d            # 4 C++ bidders + Go engine + retrainer, API on :9200
```
With no backend the page shows the recorded demo, so it still works.

## Deploy (Vercel)

```
vercel deploy --prod --yes
```
This uploads the source and builds on Vercel (not a git push; this folder is not
tracked in git). The backend IP is the default in `src/app/api/stats/route.ts`;
override with a `BACKEND_URL` env var if it changes. See `../deploy/README.md` for
starting/stopping the backend VM and re-recording the demo data.
