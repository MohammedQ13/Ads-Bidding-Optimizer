# Infrastructure Plan

## System Overview

The full system simulates a first-price RTB ad exchange.

Build status: the Go auction engine, the C++ bidders (4 strategies), the
ONNX hot-reload feedback loop, Prometheus metrics, and the Next.js console
are built and run under Docker Compose. Kafka, Redis, and Kubernetes are
the planned production target and are not wired up yet; the working MVP
uses a JSONL outcome log instead of Kafka, in-process budget state instead
of Redis, and Docker Compose instead of K8s. Those sections below are
labelled as planned where they describe components that are not built.

```
                +------------------+
                |  Go Auction Engine|
                |  (coordination)   |
                +--------+---------+
                         |  gRPC (sync, hot path)
          +--------------+--------------+
          |              |              |
  +-------v----+ +------v-----+ +-----v------+
  | C++ Bidder | | C++ Bidder | | C++ Bidder |
  | strategy A | | strategy B | | strategy C |
  +-------+----+ +------+-----+ +-----+------+
          |              |              |
          +--------------+--------------+
                         |
               +---------v----------+
               |   ONNX Runtime      |
               |   (same ML model)   |
               +--------------------+

Async feedback path:
  Auction outcomes --> Kafka --> Training pipeline --> New ONNX model
                                                          |
  C++ servers <-- K8s rolling update / hot-reload ---------+
```

**Technology choices and why:**

- C++ for inference: p99 ~2.4ms end-to-end at ~56k req/s per node, inside the
  10ms deadline, with true parallelism and zero GC pauses. The compute kernel
  alone does a bid every ~1us (~965k bids/sec).
- Go for auction engine: goroutines and channels are built for I/O-bound
  concurrency (broadcasting to multiple DSPs, enforcing timeouts,
  collecting responses).
- gRPC + protobuf for the hot path: binary serialization, sub-ms parsing,
  bidirectional streaming, native load balancing support.
- Kafka for the cold path (planned): would decouple auction outcome logging
  from model retraining so auctions keep running while the training pipeline
  consumes events at its own pace. As built, the Go engine appends outcomes
  to a JSONL file that the retrainer reads.
- Kubernetes for orchestration (planned): auto-scaling, rolling deployments
  for model updates, health-check-based routing, service discovery. As built,
  the system runs under Docker Compose; there are no K8s manifests yet.
- Prometheus + Grafana for observability: every production ML serving
  system has this. Histograms, counters, gauges on every component.
  (As built: the C++ servers still expose Prometheus `/metrics`, but instead
  of Grafana the Go engine scrapes them and aggregates everything into one
  JSON/SSE feed rendered by a custom Next.js console - see `dashboard/`.)
- Redis (planned, optional) for shared budget state across bidding instances.
  When server A spends budget, servers B and C need to know. As built, each
  bidder tracks its own budget and the Go engine paces spend per company.
- Python for offline training: PyTorch + ONNX export. Not in the hot path.

## The 10ms Constraint

The entire RTB pipeline (page load to ad rendered) happens in under
100ms. Your DSP's compute budget is roughly 10ms: receive the bid
request, encode features, run model inference, compute optimal bid,
serialize response. This constraint drives every architectural decision.

## Layer 1: ML Model (Python - current)

Already built. Produces:
- ONNX model file (bid_model.onnx) with dynamic batch sizes
- Feature config JSON (encoders, quantile arrays, tag vocab)
- Preprocessing artifacts (artifacts.pkl)

The model takes auction features and outputs a softmax probability
distribution over discrete price bins. The bid optimizer uses the
CDF (cumulative sum of bin probabilities) with the first-price profit
formula to compute the optimal bid.

## Layer 2: C++ Inference Server

### Why C++

Not because "Python is slow." Because Python cannot deliver predictable
p99 latency under concurrent load. The GIL means only one thread runs
Python at a time. Under 500 concurrent requests, threads queue and p99
becomes unpredictable. GC adds random pauses. C++ gives true parallelism,
explicit memory management with zero GC pauses, and cache-friendly data
layout.

### Service Definition (protobuf)

The full service definition (including the NotifyOutcome RPC used by the
Phase 3 feedback loop) lives in build_plan.md. The hot-path subset:

```protobuf
service BidService {
  rpc GetBid(BidRequest) returns (BidResponse);
  rpc NotifyOutcome(AuctionOutcome) returns (Ack);
  rpc Check(HealthCheckRequest) returns (HealthCheckResponse);
}

message BidRequest {
  string request_id = 1;
  int32  region = 2;
  int32  city = 3;
  string domain = 4;
  int32  ad_exchange = 5;
  int32  slot_width = 6;
  int32  slot_height = 7;
  int32  slot_visibility = 8;
  int32  slot_format = 9;
  string advertiser_id = 10;
  repeated string user_tags = 11;
  float  slot_floor_price = 12;
  float  impression_value = 13;   // V, set by Go engine per auction
  int64  timestamp = 14;
}

message BidResponse {
  string dsp_id = 1;              // which "company" this is
  float  bid_price = 2;
  float  win_probability = 3;     // P(win at bid_price)
  float  expected_profit = 4;     // (V - bid) * P(win)
  bool   used_fallback = 5;
  int64  inference_us = 6;        // inference duration microseconds
}
```

The predicted bin distribution stays server-side. The C++ server computes
the bid from the CDF and returns just the price plus a few diagnostics.
Less data over the wire, faster.

### Component 1: Fixed Thread Pool

Pre-allocate N worker threads at startup (typically 2x CPU cores).
Fixed-size is deliberate - dynamic pools allocate memory at request
time, adding latency jitter. Each worker has a pre-allocated feature
vector buffer. Workers pull from a bounded work queue. When the queue
is full, new requests get RESOURCE_EXHAUSTED (load shedding).

### Component 2: Request Deduplication (LRU Cache)

Networks are unreliable, clients retry. Without dedup, a retried request
runs inference twice. Fixed-size LRU cache keyed on request_id with
500ms TTL. Cache hit returns stored BidResponse immediately.

### Component 3: In-Memory Feature Store

Pre-computed feature encodings loaded at startup from feature_config.json:
- Categorical label encoders in std::unordered_map
- Quantile transformer arrays (binary search + interpolation + erfinv)
- Tag vocabulary lookup
- All hash maps pre-reserved to avoid rehashing

For unknown entities (new domain, unseen user segment), fall back to
default embedding. Log cold-start lookups to Prometheus.

### Component 4: Micro-Batcher

Individual ONNX Runtime calls have fixed overhead (session setup, tensor
allocation, kernel dispatch). A batch of 32 takes roughly the same time
as a batch of 1, making per-item throughput 32x better.

The micro-batcher accumulates requests and flushes when batch reaches
size N or timeout T expires, whichever comes first. Results dispatched
back via per-request promise/future pairs.

Tunable tradeoff:
- Small T, small N: low latency, low throughput
- Large T, large N: high throughput, higher latency

Benchmark both to find the Pareto frontier.

### Component 5: ONNX Runtime Inference

Load ONNX model at startup into an InferenceSession. Sessions are
thread-safe for concurrent reads. Each worker creates its own
input/output tensor buffers. Input: [batch, feature_dim].
Output: [batch, num_bins] softmax probabilities over price bins.

After inference, each worker computes the CDF (cumulative sum of bin
probabilities) and finds the optimal bid using ternary search on the
first-price profit function (~30 iterations, nanoseconds).

### Component 6: Circuit Breaker

Three states: CLOSED (normal), OPEN (inference failing, use fallback),
HALF-OPEN (cooldown expired, test one request). If the test succeeds,
back to CLOSED. Prevents cascading failures when ONNX Runtime has issues.

### Component 7: Fallback Bid

When circuit is OPEN or SLA budget exhausted: return a pre-computed
heuristic bid (historical average clearing price * 0.7 shading factor).
The DSP always participates in auctions even under degraded conditions.
Track fallback rate in Prometheus.

### Component 8: Graceful Shutdown

On SIGTERM: stop accepting new connections, drain in-flight requests
(5 second timeout), then exit. No in-flight bid computations killed.

### Component 9: Prometheus Observability

Expose /metrics HTTP endpoint. Instrument:
- request_total: counter by status (success, timeout, rejected, fallback)
- request_latency_seconds: histogram (1ms, 2ms, 5ms, 10ms, 25ms, 50ms buckets)
- inference_latency_seconds: ONNX Runtime only
- feature_lookup_latency_seconds: feature store lookups
- thread_pool_active_workers: gauge
- request_queue_depth: gauge
- batch_size: histogram of actual batch sizes flushed
- circuit_breaker_state: gauge (0=closed, 1=open, 2=half-open)
- cache_hit_total: counter for dedup cache hits
- cold_start_lookup_total: counter for unknown entities

### Component 10: Health Check

Standard gRPC health check protocol. SERVING when model loaded and
feature store ready. NOT_SERVING during startup. Load balancers use
this to route traffic only to ready instances.

### Config File

All tunable parameters loaded from YAML at startup:
- thread_pool_size, batch_max_size, batch_timeout_us
- sla_budget_us, circuit_breaker_threshold, circuit_breaker_cooldown_ms
- dedup_cache_size, dedup_cache_ttl_ms
- fallback_bid_by_category
- model_path, feature_store_path

## Layer 3: Go Auction Engine

### What It Does

Simulates the exchange/SSP side. For each auction:
1. Read auction record from dataset (or generate from simulation)
2. Broadcast BidRequest to C++ DSP via gRPC
3. Generate bids from N calibrated synthetic competitors
4. Enforce 10ms deadline - late DSPs excluded
5. Run first-price auction - highest bid wins, winner pays their bid
6. Notify C++ DSP of win/loss and clearing price
7. Log auction outcome for retraining

### Calibrated Synthetic Competitors

Simulated competitors per auction, set by `competitors.count`. The shipped
`go-engine/config.yaml` runs 8 competitors (the code falls back to 4 if the
config is missing the field). Each competitor draws a bid from a lognormal
centered on the base median clearing price (`competitors.base_median`, 50 fen
in the shipped config, 75 if unset), with sigma `competitors.spread` (0.5 in
the shipped config, 0.6 if unset), then scales the draw by an archetype
multiplier and floors it at 1 fen. The pool cycles through eight archetypes
so a larger field still looks varied:

| Archetype | Multiplier |
|-----------|------------|
| whale | 1.6 |
| aggressive | 1.4 |
| sniper | 1.25 |
| contender | 1.15 |
| market | 1.0 |
| value | 0.85 |
| conservative | 0.7 |
| bargain | 0.55 |

Tuning the count, base median, and spread sets how hard the field is to
beat. The shipped values (8 competitors, median 50, spread 0.5) produce a
balanced market where all four C++ companies win some auctions rather than
one running away with it.

### Budget Management

Each advertiser has a daily budget. The Go layer tracks spend and
adjusts a bid multiplier. When budget is running low, servers bid
more conservatively. Budget pacing spreads spend evenly over time.

### Impression Value (V)

V is per-auction, computed by the Go layer and passed in BidRequest.
In a full system: V = pCTR * advertiser_CPA. For simulation, V comes
from config per advertiser. The ML model never sees V - it only
predicts clearing price distribution. V only matters for bid optimization.

## Multi-Instance Bidding Strategies

This is where the distributional prediction pays off for the distributed
system. A point estimate model outputs one number - every server computes
the same bid. Nothing to compare. A distributional model outputs a full
probability curve, and different servers can apply different strategies
on top of the same distribution:

- **Server A (profit maximizer):** default strategy, picks the bid where
  (V - b) * P(win) is maximized. Balanced risk/reward.
- **Server B (aggressive):** applies a 1.2x multiplier to the optimal bid.
  Wins more auctions, lower margin per win. Good when budget is flush.
- **Server C (conservative):** 0.8x multiplier. Wins fewer auctions but
  only the cheap ones. High margin per win. Good for tight budgets.
- **Server D (budget-paced):** starts aggressive early in the day, gets
  more conservative as daily budget drains. Multiplier decreases over time.
- **Server E (percentile bidder, optional 5th strategy):** bids at the
  70th percentile of the predicted distribution. High win rate, ignores
  the profit curve. Added if time permits - the core demo runs A-D.

All servers run the same ONNX model, get the same mixture distribution,
but produce different bids. The Go auction engine pits them against each
other AND against synthetic competitors. You can measure:
- Which strategy wins the most auctions
- Which strategy generates the most total profit
- Which strategy uses budget most efficiently
- How strategies perform under different market conditions

This is a live multi-strategy comparison rather than a classic A/B test:
the strategies run as separate server instances against the same market and
the same distributional predictions, so their profit and win-rate can be
compared head to head, which is how production DSPs evaluate changes.

The model also serves a second purpose: the Go layer can SAMPLE from the
predicted distribution to generate realistic synthetic competitor bids.
The model becomes both the bidder and the source of competitor behavior.
No hand-tuning competitor distributions needed.

## Closed-Loop Simulation

This is what makes the project different from static ML work. The model
learns from its own operating environment using a sliding window of
recent auction outcomes.

1. Pretrain: train initial model on iPinYou historical data (the warm
   start - gives the model real auction priors)
2. Simulate: auction engine generates traffic, DSP bids against
   competitors (synthetic + other C++ server instances)
3. Collect: the Go engine appends each auction outcome to a JSONL log
   (Kafka in the planned version). Each record is a labeled example:
   auction features plus the actual clearing price.
4. Sliding window: training pipeline maintains a buffer of the last W
   outcomes (e.g. 500K). As new data arrives, oldest data drops out.
   This prevents synthetic data from accumulating indefinitely - the
   model always trains on the most recent competitive environment.
5. Fine-tune: Python script loads current checkpoint, runs a few NLL
   epochs on the sliding window, validates calibration on held-out 10%.
6. Pointer swap: export new ONNX to shared volume. Each C++ server
   detects the new file, loads it into a second ONNX session on a
   background thread, validates outputs (no NaN, reasonable ranges),
   then atomically swaps the session pointer
   (std::atomic<shared_ptr<Session>>). In-flight requests finish on the
   old model, new requests use the new model. Zero downtime.
7. Observe: the Next.js console (fed by Prometheus scrapes) shows whether
   metrics improve after each swap. If not, roll back to the previous
   checkpoint.

The sliding window means the model always reflects current conditions.
It never drifts into training purely on synthetic data - the window
size bounds how much simulation data the model sees at any point, and
the iPinYou-trained checkpoint provides the starting prior that grounds
the model in real auction dynamics.

## Kafka Event Pipeline (planned)

Status: not built. The MVP uses a single JSONL outcome log written by the Go
engine and read by the retrainer. The two-topic design below is the planned
production version.

Two Kafka topics:

**auction-outcomes:** published by Go auction engine after every auction.
Contains: auction features, all bids, winner, clearing price, your DSP's
bid and profit/loss. Training pipeline consumes this for model retraining.
Also consumed by analytics for offline reporting.

**bid-requests:** published by C++ servers for every bid computed. Contains:
request features, predicted distribution params, computed bid, latency.
Used for auction replay (test new strategies on historical requests
without running live auctions) and debugging.

Kafka decouples producers from consumers. Auctions run at full speed
regardless of how fast the training pipeline processes events. Events
are retained for replay. Multiple consumers can read the same stream.

## Kubernetes Deployment (planned)

Status: not built. The working system runs under Docker Compose
(`docker-compose.yml`); there are no K8s manifests in the repo yet. The
layout below is the production target.

```
Namespace: rtb-system

Deployments:
  cpp-bidder        (4 replicas, HPA on CPU)
  go-auction-engine (1 replica)
  training-worker   (1 replica, CronJob or triggered)

StatefulSets:
  kafka             (3 brokers)
  redis             (1 replica, for budget state)

Services:
  cpp-bidder-svc    (ClusterIP, gRPC load balancing)
  kafka-svc         (Headless)
  redis-svc         (ClusterIP)

Monitoring:
  prometheus        (scrapes all cpp-bidder pods)
  grafana           (dashboards)
```

Why Kubernetes over Docker Compose:
- HPA (Horizontal Pod Autoscaler) scales bidder replicas based on QPS
- Rolling updates for model hot-reload: new ONNX version rolls out pod
  by pod, with health checks ensuring each pod is ready before moving on
- Liveness/readiness probes use the gRPC health check endpoint
- Service discovery: Go engine finds bidders through K8s DNS
- Resource limits prevent a single pod from starving others
- ConfigMaps for server config, Secrets for any credentials

Docker Compose is still useful for local dev and quick demos. The
Kubernetes manifests are the production deployment target.

## Budget State with Redis (planned)

Status: not built. As built, each bidder tracks its own budget and the Go
engine paces per-company spend, which is enough for the single-process demo.
Shared Redis state matters once the same advertiser budget is split across
multiple bidder replicas.

Multiple bidding servers spend from the same advertiser budget. Without
shared state, each server tracks budget independently and the total
spend overshoots.

Redis holds per-advertiser budget counters:
- C++ server does DECRBY after winning an auction
- Go engine checks remaining budget before sending bid requests
- Atomic operations prevent race conditions
- TTL-based daily budget reset

Redis adds ~0.1ms per call, well within the 10ms budget. Only called on
wins (not every bid request), so the load is manageable.

Alternative: a dedicated Go budget service with in-memory state and
gossip protocol for multi-node sync. More complex but avoids the Redis
dependency. Either approach works.

## Benchmarks (5 specific)

1. Thread pool scaling: QPS vs pool size at fixed concurrency
2. Latency vs concurrency: p50/p95/p99 at 50/100/200/500/800 clients
3. Micro-batcher tuning: throughput vs p99 across batch size and timeout configs
4. Bid quality: bid regret on held-out data - MDN vs point estimate vs naive
5. Strategy comparison: total profit across 100K auctions for each
   bidding strategy (aggressive, conservative, profit-max, percentile, paced)

Use ghz for gRPC load testing. Always warm up with 10K requests before
benchmarking (ONNX Runtime JIT compilation).
