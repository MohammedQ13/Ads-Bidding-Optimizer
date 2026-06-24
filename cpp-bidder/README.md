# C++ Bid Server (Phase 2)

A gRPC inference server that turns the Phase 1 price-distribution model into
live bids. The Go auction engine (Phase 3) calls `GetBid` for every auction; the
server encodes the request features, runs the ONNX model, turns the predicted
price distribution into a bid, and returns it under a tight latency budget.

The same binary runs as any "company" (a strategy) by pointing it at a different
config file. Four companies compete in the auction with the same model but
different bid multipliers.

## What it does

```
BidRequest (gRPC)
  -> feature encoding        9 categoricals + 9 continuous + up to 10 tags
  -> dedup cache check       same request_id within 500ms returns cached bid
  -> circuit breaker check   if the model is failing, serve the fallback bid
  -> micro-batcher           groups requests, one ONNX Run() per batch
  -> ONNX inference          301-bin softmax over integer fen prices (bin k = k fen)
  -> bid optimizer           argmax of (V - b) * CDF(b)
  -> BidResponse (gRPC)
```

The bid math matches the Python evaluation pipeline exactly (bin index == price
in fen, `grid_optimize_bins`), so the server reproduces the regret numbers the
model was selected on. This is enforced by a parity test.

## Layout

```
proto/bid_service.proto      gRPC contract (GetBid / NotifyOutcome / Check)
src/
  feature_store.*            raw request -> model tensors (matches training)
  model_session.*            ONNX Runtime wrapper + atomic hot-reload seam
  bid_optimizer.*            (V - b) * CDF(b) grid optimizer
  micro_batcher.*            opportunistic batching, bounded work queue
  dedup_cache.*              LRU + TTL keyed on request_id
  circuit_breaker.*          CLOSED / OPEN / HALF-OPEN around inference
  fallback.*                 model-free bid when degraded
  metrics.*                  Prometheus /metrics endpoint
  config.*                   YAML config loader
  bid_service.*              the gRPC service, wires it all together
  main.cpp                   startup, health, graceful shutdown, model watcher
test/
  generate_fixture.py        builds the golden fixture from the Python pipeline
  parity_test.cpp            C++ hot path vs the golden fixture
  grpc_smoke.cpp             end-to-end GetBid over gRPC vs the fixture
  bench.cpp                  load generator (throughput + latency percentiles)
models/                      bid_model.onnx + feature_config.json
config/                      config.yaml (company A) + company_{b,c,d}.yaml
monitoring/                  Prometheus scrape config + Grafana dashboards
```

## Build and test

Everything builds in Docker so the gRPC / protobuf / ONNX Runtime / prometheus-cpp
toolchains do not have to be solved locally.

```
# build the server image (compiles bid_server, parity_test, grpc_smoke, bench)
docker build -t cpp-bidder:latest .

# hot-path parity test (no gRPC, fast)
docker build -f Dockerfile.test -t cpp-bidder-parity .

# end to end: start a server, replay the fixture over gRPC, check metrics
docker run -d --name bidder cpp-bidder:latest
docker exec bidder ./build/grpc_smoke localhost:50051 test/golden_fixture.json
docker exec bidder sh -c "wget -qO- http://localhost:9100/metrics | grep '^bid_'"
```

To regenerate the golden fixture after a model re-export:

```
python test/generate_fixture.py
```

## Run the full system

```
docker compose up --build
```

Brings up the four DSP companies (A profit-max, B aggressive, C conservative,
D budget-paced), Prometheus, and Grafana. Grafana is on `:3000` (anonymous admin)
with the "RTB Bidder Scoreboard" dashboard provisioned. This compose file is the
standalone C++-only dev rig.

For the **full system** (the C++ fleet, the Go auction exchange, and the Python
retrainer wired into the feedback loop) use the root `docker-compose.yml` and the
custom telemetry console in `dashboard/` (the engine scrapes these servers'
`/metrics` and aggregates everything into one live JSON/SSE feed). The Go engine
that drives traffic now lives in `go-engine/`.

gRPC ports: A `50051`, B `50052`, C `50053`, D `50054`.
Metrics ports: A `9101`, B `9102`, C `9103`, D `9104`.

## Design notes

The deep design (threading, the ONNX oversubscription trap, micro-batching
tradeoffs, hot-reload, backpressure, tail latency, scaling) is in
`../rtb-bid-model/docs/cpp_server.md`. Two implementation choices worth calling
out:

- **Sync gRPC, not async completion queues.** gRPC's synchronous API with its
  own thread pool gives request concurrency, and the micro-batcher amortizes
  inference. This is simpler and less bug-prone than hand-rolled async CQ code.
  The async API is the documented upgrade path if a benchmark shows the sync
  thread pool is the bottleneck.
- **ONNX single-threaded.** `intra_op = inter_op = 1`. The model is tiny, so
  intra-op parallelism is pure overhead; concurrency comes from many workers
  each running a sequential forward pass on the shared, thread-safe session.

## Status

Phase 2 is functionally complete: feature parity with Python, ONNX inference,
bid optimizer, micro-batcher, dedup cache, circuit breaker, fallback, Prometheus
metrics, gRPC health, graceful shutdown, YAML config, and the hot-reload seam.
The model file watcher is wired but off by default (Phase 3 turns it on for the
retraining loop). Benchmarks live in `test/bench.cpp`.
