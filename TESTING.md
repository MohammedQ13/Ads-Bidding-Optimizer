# Testing & Status

What is tested, how to run it, and what is left.

## Test layers

| Layer | Type | Where | Runs via | Status |
|-------|------|-------|----------|--------|
| C++ feature encoding + bid math | parity vs Python | `cpp-bidder/test/parity_test.cpp` | Docker build / CTest | PASS |
| C++ gRPC path end-to-end | integration | `cpp-bidder/test/grpc_smoke.cpp` | `docker exec` | PASS |
| C++ components (breaker, cache, optimizer) | unit | `cpp-bidder/test/unit_test.cpp` | CTest | PASS |
| C++ load/latency | benchmark | `cpp-bidder/test/bench.cpp` | manual | done (docs/benchmarks.md) |
| Go engine logic | unit + integration (mock DSP) | `go-engine/internal/**/**_test.go` | `go test ./...` | PASS (runs in image build) |
| Python retrainer encoding/export | unit | `retrainer/test_retrain.py` | `pytest` | todo |
| Full feedback loop | live e2e | `scripts/live_e2e.sh` | private docker net | PASS (6/6 assertions) |
| Strategy performance | backtest on real data | `rtb-bid-model/src/backtest.py` | `python src/backtest.py` | PASS (found + fixed a real bug) |

## Bug the backtest caught (and the fix)

The backtest measured profit-max regret at ~30 fen, not the documented ~20. Root
cause: the deployed model was `bins_quant200`, which is **quantile-spaced** - the
bin index is NOT the price in fen, it maps through an `edges` array. But the C++
optimizer, the parity fixture, the backtest, and the retrainer all assume bin
index == price (uniform). The edges were never exported, so every bid was
computed against the wrong price mapping.

Fix: deploy the **uniform** `bins_300` model (bin k = k fen, full-set regret
20.327, matching the record). No C++ code change; the optimizer reads `num_bins`
from the model shape and scans generically. After the swap: profit-max regret
19.57 (200k sample), C++ parity still passes, fixture regenerated. This is exactly
what a backtest is for.

## How to run

```
# C++ unit + parity tests
docker build -t cpp-bidder:latest cpp-bidder
docker run --rm cpp-bidder:latest sh -c "cd build && ctest --output-on-failure"

# Go tests (also run automatically during the engine image build)
docker build -t rtb-engine:latest go-engine     # fails if go test fails

# live end-to-end (boots the stack, asserts the loop, tears down)
bash scripts/live_e2e.sh
```

## What "tested" means here

- **Unit:** one component in isolation, deterministic, no network. Fast.
- **Integration (mock):** the engine driven against a fake DSP client so we test
  auction logic (first-price, deadline, budget) without standing up C++ servers.
- **Live e2e:** the real stack in Docker (companies + engine + retrainer) with
  assertions that auctions flow, outcomes log, and the model hot-swaps.
- **Backtest:** replay real iPinYou auctions (features + the real clearing price)
  through each strategy and report profit/regret. This grounds the synthetic
  simulation against reality.

## Visualization (built)

We dropped Prometheus + Grafana (already used elsewhere) for a custom telemetry
console. The Go engine is the single aggregation point: it exposes one rich JSON
snapshot plus an SSE stream, and a Next.js dashboard (`dashboard/`) renders the
whole distributed system in real time.

- `GET /stats`  -> JSON snapshot: per-strategy economics + model output (P(win),
  expected vs realized profit) + round-trip and ONNX latency percentiles, plus
  system-wide histograms, the impression mix, the competitor archetypes, a live
  auction feed, the scraped C++ internals, and the retrainer status.
- `GET /events` -> Server-Sent Events stream of that snapshot (~2 Hz).

The engine also scrapes each C++ server's Prometheus `/metrics` (breaker, queue,
batch, cache, cold-start, model version) and reads the retrainer's status file,
folding both into the same feed. The dashboard is a dark "mission control"
console with panels for every component: topology, KPIs, latency budget, the
profit race, strategy cards, the live auction stream, model predictions, the
market mix, the C++ internals table, and the retraining loop.

**Screenshot verification:** `dashboard/shot.mjs` + `shot-detail.mjs` drive a
headless Chromium (Playwright) against the live stack and capture the console in
both themes, used to verify every panel renders real data end to end.

## Performance (benchmarks)

These measure three different things at three layers, so they don't conflict (full
explanation in `cpp-bidder/docs/benchmarks.md`):
- **Compute**: `bench_engine` (in-process, no network): **~965k bids/sec on a
  compute-optimized 16-vCPU c2d-standard-16 node = ~1.0µs/bid** (full pipeline: encode
  + ONNX inference + first-price optimize). The throughput and the per-bid latency are
  the same fact (1 / 965k s ≈ 1.0µs). A budget e2-standard-16 (shared cores) gives ~620k
  on the same binary.
- **End-to-end serving**: `bench` / `bench_async` (over the async gRPC server):
  **~56k req/s at p99 = 2.4ms** on cloud (well inside the 10ms auction deadline);
  ~79k req/s peak. The p99 is dominated by network/protobuf/scheduling around the
  ~1.0µs of compute, not the model. WSL2 caps the laptop at ~6-8k req/s.
- **Demo pace**: the dashboard market runs at ~200 auctions/sec on purpose
  (`rate_per_sec`), a readability/cost knob, not a capacity limit.

## Scoreboard balance

Budget pacing (binding daily budgets that reset per "day") + competitor
calibration give a balanced, differentiated scoreboard: profit-max best profit,
aggressive most volume / thin margin, conservative selective / high margin,
budget-paced ~ profit-max. Tuned in `go-engine/config.yaml`, deterministic unit
test in `engine_test.go` (`TestBudgetResetsPerDay`).

## Left to do

- [x] C++ unit tests, Go test suite, retrainer unit tests
- [x] Engine JSON/SSE stats API; removed Prometheus/Grafana
- [x] Scripted live e2e (`scripts/live_e2e.sh`), passes 6/6
- [x] Backtest on real iPinYou test data (found + fixed the quantile-bins bug)
- [x] cpp-bidder image rebuilt with corrected model; all images current
- [x] Engine replay mode CLI flag + benchmarks (`bench_engine`, `bench_async`)
- [x] Budget pacing + competitor calibration (balanced scoreboard)
- [x] Async completion-queue gRPC server (built + verified; handles high
      concurrency where sync collapsed; absolute QPS env-bound on WSL2)
- [x] Enriched engine telemetry (ML output, latency percentiles, histograms,
      live auction feed, scraped C++ internals, retrainer status)
- [x] Custom Next.js telemetry console consuming the JSON/SSE API, screenshot-
      verified end to end against the live Docker stack (dark + light)
