# Build Plan -- Phases 2-4

## The Big Picture

```
  DSP "Company A"  DSP "Company B"  DSP "Company C"  DSP "Company D"
  (aggressive)     (conservative)   (profit-max)     (budget-paced)
       |                |                |                |
       +-------+--------+--------+-------+
               |        gRPC     |
        +------v-----------------v------+
        |      Go Auction Engine        |
        |      (the exchange/SSP)       |
        +------+------------------------+
               |
               v
        Auction Outcomes (batched)
               |
               v
        +------+--------+
        | Kafka / Queue  |
        +------+---------+
               |
               v
        Python Retraining Worker
        (fine-tune on recent outcomes)
               |
               v
        New ONNX model
               |
               v
        Hot-reload into all DSP pods
        (atomic session swap, zero downtime)
```

Each C++ instance is a separate "company" competing for ad space.
Same model, different strategies. The Go engine is the exchange that
runs the auction. Outcomes feed back into retraining. The model gets
better over time by learning from its own competitive environment.

---

## Model Deployment Decision

The ML phase produced two final results: a single model (bins_quant200,
20.17 fen regret) and an ensemble of 5 models (19.908 fen). The C++
server needs to load and run one of these. Three options were considered.

### Option A: Single model (chosen)

Export bins_quant200 as one ONNX file. The C++ server loads it, runs
inference, returns the 200-bin probability distribution to the Go
engine, which computes the optimal bid.

Pros:
- One ONNX file, simple to load and swap on retrain
- Fast inference (single forward pass)
- Hot-reload is straightforward: load new file, validate, pointer swap
- The feedback loop will retrain a single model anyway, so the ensemble
  advantage disappears once the system is running live

Cons:
- 0.26 fen worse than the ensemble on the static iPinYou test set

### Option B: Ensemble in C++ (not chosen)

Export all 5 models as separate ONNX files. C++ loads each one, runs
inference on all 5, blends their output distributions with the ensemble
weights (49 mdn_s42, 5 mdn_c1, 4 mdn_c4, 40 bins_quant200, 2
bins_sqrt300), and returns the blended distribution.

Pros:
- Best static accuracy (19.908 fen)

Cons:
- 5x inference cost and memory
- MDN outputs are Gaussian mixture parameters, not bin probabilities.
  The C++ server would need to convert MDN outputs to discrete bins
  before blending, adding complexity and a potential source of bugs.
- Hot-reload becomes harder: swapping one member means re-validating
  the whole ensemble, and the feedback loop would need to decide which
  member(s) to retrain
- The 0.26 fen gap is small and disappears once the model retrains on
  live auction outcomes

### Option C: Bake ensemble into one ONNX (not chosen)

Write a PyTorch wrapper that loads all 5 models internally, does the
blending in the forward pass, and exports the whole thing as a single
ONNX file. The C++ server sees one model.

Pros:
- Best accuracy with simple serving interface

Cons:
- Larger ONNX file (~7MB vs ~1.4MB)
- MDN-to-bins conversion baked into the ONNX graph adds complexity
  to the export and makes debugging harder
- Cannot swap individual ensemble members on retrain. Any model update
  requires re-exporting the entire wrapper, which defeats the purpose
  of incremental retraining in the feedback loop.

### Why Option A

The ensemble's 0.26 fen advantage is measured on the static iPinYou
test set with a temporal distribution shift. Once the feedback loop is
running, the model retrains on recent auction outcomes from its own
competitive environment. At that point it produces a single updated
checkpoint, not an ensemble. The ensemble is a Phase 1 artifact that
maximized offline performance; it does not carry forward into production.

Starting with a single model also keeps the C++ server simple. The hot
path (feature encoding, ONNX inference, bid optimization) is already
nontrivial. Adding ensemble blending and MDN-to-bins conversion in C++
would increase development time and surface area for bugs, for a gain
that the feedback loop erases.

If the feedback loop is descoped or delayed, Option C becomes the
pragmatic fallback: re-export the ensemble as a single ONNX and get
19.908 fen with no C++ changes.

---

## Phase 2 -- C++ Inference Server (May-Jul 2026)

### Week 1: Docker + dependencies + hello world

Get the hardest part done first. gRPC + protobuf + ONNX Runtime each
have their own CMake setup and they conflict. Solve it with Docker.

- [ ] Dockerfile: Ubuntu base, install gRPC, protobuf, ONNX Runtime,
      prometheus-cpp, yaml-cpp
- [ ] CMakeLists.txt that links everything
- [ ] Write bid_service.proto (BidRequest/BidResponse)
- [ ] Bare BidService::GetBid that returns a hardcoded bid
- [ ] Verify it compiles and runs in container
- [ ] docker-compose.yml with just the one service for now

Proto definition:

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
  float  win_probability = 3;
  float  expected_profit = 4;
  bool   used_fallback = 5;
  int64  inference_us = 6;
}

message AuctionOutcome {
  string request_id = 1;
  bool   won = 2;
  float  clearing_price = 3;      // winning bid amount
  float  your_bid = 4;
  float  profit = 5;              // (V - bid) if won, 0 if lost
}
```

### Week 2-3: Core hot path

End-to-end: BidRequest -> features -> ONNX -> bid -> BidResponse.

- [ ] Feature store: load feature_config.json at startup
  - categorical encoders in unordered_map (pre-reserved)
  - QuantileTransformer: binary search on exported quantile arrays,
    interpolate, apply erfinv for normal output
  - tag vocab lookup
  - unknown entities -> default embedding, log cold-start counter
- [ ] ONNX Runtime inference
  - load bid_model.onnx into InferenceSession
  - input: [batch, 9 cats] + [batch, 9 conts] + [batch, tag_seq_len]
  - output: [batch, num_bins] softmax probabilities over price bins
- [ ] Bid optimizer in C++
  - CDF: cumulative sum of bin softmax probabilities
  - ternary search on (V - b) * CDF(b) for optimal bid (~30 iterations)
  - nanoseconds per bid vs grid search microseconds
- [ ] Fixed thread pool
  - pre-allocated N workers (2x CPU cores)
  - pre-allocated feature vector buffers per worker
  - bounded work queue, RESOURCE_EXHAUSTED when full

### Week 4-5: Reliability layer

- [ ] Micro-batcher
  - accumulate requests, flush on batch_size or timeout
  - dedicated batching thread
  - per-request promise/future dispatch
  - configurable N (batch size) and T (timeout)
- [ ] LRU dedup cache
  - keyed on request_id, 500ms TTL
  - cache hit returns stored BidResponse immediately
  - fixed size, no memory growth
- [ ] Circuit breaker
  - CLOSED -> OPEN after N consecutive failures
  - OPEN -> HALF-OPEN after cooldown
  - HALF-OPEN -> CLOSED on success, OPEN on failure
  - track state transitions in Prometheus
- [ ] Fallback bid
  - historical average clearing price * configurable shading factor
  - per ad-category fallback from config
  - always participate, even when degraded

### Week 6: Observability + ops

- [ ] Prometheus /metrics endpoint
  - request_total (counter by status)
  - request_latency_seconds (histogram: 1,2,5,10,25,50ms buckets)
  - inference_latency_seconds
  - feature_lookup_latency_seconds
  - thread_pool_active_workers (gauge)
  - request_queue_depth (gauge)
  - batch_size (histogram)
  - circuit_breaker_state (gauge: 0/1/2)
  - cache_hit_total (counter)
  - cold_start_lookup_total (counter)
- [ ] gRPC health check (SERVING / NOT_SERVING)
- [ ] Graceful shutdown (SIGTERM -> drain -> exit)
- [ ] Config file loading (YAML)
  - thread_pool_size, batch_max_size, batch_timeout_us
  - sla_budget_us, circuit_breaker_threshold
  - dedup_cache_size, dedup_cache_ttl_ms
  - fallback_bid_by_category
  - model_path, feature_store_path
  - dsp_id, strategy_type, bid_multiplier

### Week 7: Benchmarks

- [ ] Benchmark 1: thread pool scaling
  - QPS vs pool size (1, 2, 4, 8) at 200 concurrent clients
  - should scale linearly until CPU saturation
- [ ] Benchmark 2: latency vs concurrency
  - p50/p95/p99 at 50/100/200/500/800 clients
  - flat latency until saturation, then graceful degradation
- [ ] Use ghz for gRPC load testing
- [ ] Warm up with 10K requests before benchmarking (ONNX JIT)

---

## Phase 3 -- Go Auction Engine + Feedback Loop (Aug-Sep 2026)

### Week 1-2: Multi-DSP auction engine

The Go engine IS the exchange. It runs first-price auctions where
multiple C++ DSP instances compete against each other.

- [ ] Go gRPC client that broadcasts BidRequest to N DSP instances
- [ ] Each DSP instance configured as a different "company":
  - Company A: profit maximizer (bid_multiplier=1.0)
  - Company B: aggressive (bid_multiplier=1.2, wins more, lower margin)
  - Company C: conservative (bid_multiplier=0.8, wins less, higher margin)
  - Company D: budget-paced (multiplier decreases over simulated day)
- [ ] 10ms deadline enforcement: late DSPs excluded from auction
- [ ] First-price auction logic: highest bid wins, pays their bid
- [ ] Win/loss notification back to each DSP via NotifyOutcome RPC
- [ ] Auction logging: all bids, winner, clearing price, profit/loss
- [ ] Budget management per company (daily budget, spend tracking)
- [ ] Impression value (V) per auction from config or computed

### Week 3: Sliding window feedback loop

Batch auction outcomes and feed them back into the model. The model
retrains on a sliding window of recent outcomes, so it learns the
current competitive environment instead of stale historical data.

```
Auction outcomes stream in
        |
        v
Batch buffer (accumulate N outcomes, e.g. 50K)
        |
        v
Flush to Kafka topic: "auction-outcomes"
        |
        v
Python retraining worker consumes from Kafka
        |
        v
Sliding window: keep last W outcomes (e.g. 500K)
Drop oldest data as new data arrives
        |
        v
Load current checkpoint (initially iPinYou-trained model)
Fine-tune: few epochs of NLL on the sliding window
Validate: calibration check on held-out 10% of window
        |
        v
Export new ONNX model to shared volume
        |
        v
C++ servers detect new model file (file watcher)
        |
        v
Each server:
  1. Background thread loads new ONNX into second InferenceSession
  2. Validate outputs (no NaN, reasonable ranges)
  3. Atomic pointer swap (std::atomic<shared_ptr<Session>>)
  4. New requests use new model immediately
  5. Old model freed when last in-flight request finishes
        |
        v
Grafana shows whether metrics improved after swap
If regret worsens -> rollback to previous checkpoint
```

- [ ] Kafka setup (or simple file-based batching for MVP)
  - topic: auction-outcomes (features + all bids + clearing price)
  - topic: bid-requests (for replay/debugging)
- [ ] Python retraining script
  - consumes from Kafka into a sliding window buffer
  - window size W configurable (default 500K outcomes)
  - loads current checkpoint, fine-tunes on window
  - validates on held-out slice of window
  - exports new ONNX if validation improves
  - saves checkpoint with timestamp for rollback
- [ ] Pointer swap in C++ server
  - file watcher on model directory (inotify / polling)
  - background thread loads new ONNX into fresh InferenceSession
  - validates: run 100 cached inputs, check no NaN, output shape correct
  - atomic swap: std::atomic<shared_ptr<Ort::Session>>
  - old session ref-counted, freed after in-flight requests drain
  - zero downtime, zero dropped requests
- [ ] Drift detection
  - monitor val NLL on sliding window
  - if degrading, trigger retraining automatically

### Week 4: Docker Compose multi-instance

- [ ] docker-compose.yml:
  - 4x C++ DSP instances (companies A-D)
  - 1x Go auction engine (the exchange)
  - 1x Kafka (event pipeline)
  - 1x Python retraining worker (sliding window from Kafka)
  - 1x Prometheus (scrapes all DSP pods)
  - 1x Grafana (dashboards)
- [ ] Nginx or Envoy in front of DSPs? No -- Go engine calls each
      DSP directly since they're separate companies, not load-balanced
      replicas of the same service
- [ ] Benchmark 3: micro-batcher tuning
  - throughput vs p99 across batch size and timeout configs
- [ ] Strategy comparison benchmark
  - total profit per company across 100K auctions
  - win rate per company
  - budget efficiency per company

---

## Phase 4 -- Polish + Demo (Oct-Nov 2026)

### Week 1-2: Grafana dashboard (the demo artifact)

Real-time panels updating while auctions run:

- [ ] QPS gauge across all DSP instances
- [ ] Latency percentiles (p50/p95/p99) per instance
- [ ] Win rate per company (trailing 1000 auctions)
- [ ] Profit per won auction per company
- [ ] Cumulative profit per company (the scoreboard)
- [ ] Budget utilization per company
- [ ] Model version per instance
- [ ] Circuit breaker state per instance
- [ ] Fallback rate
- [ ] Batch retraining events (when model updates happen)

### Week 2: Bid landscape visualization

- [ ] For a given auction: show predicted mixture PDF, actual clearing
      price, each company's bid as colored vertical lines
- [ ] Shows how different strategies interpret the same distribution

### Week 3: Final benchmarks + documentation

- [ ] All 4 benchmarks with final numbers
- [ ] Architecture diagram
- [ ] README with setup instructions and results
- [ ] Record a demo video: start docker-compose, show Grafana updating
      in real time, show model hot-reload happening

### Week 4: Buffer

For anything that took longer than expected.

### Optional enhancements (if time permits)

- [ ] K8s manifests (HPA, rolling updates, health probes)
- [ ] Redis for shared budget state across instances
- [ ] Per-company model variants (different training data or architecture)
- [ ] Auction replay: test new strategies on historical auction logs
- [ ] pCTR model for dynamic impression value (V)

---

## Risk Mitigation

1. **C++ dependency hell (Phase 2 Week 1):** Docker solves this.
   Never fight local deps. Write the Dockerfile first.

2. **QuantileTransformer in C++:** Arrays are already exported in
   feature_config.json. Implement: binary search on quantiles array,
   interpolate between nearest quantiles, apply erfinv. Test against
   Python output on 1000 samples.

3. **Scope creep:** The C++ server with Prometheus alone is already
   portfolio-worthy. Feedback loop is enhancement. If Phase 3 runs
   long, ship without it.

4. **Hot-reload correctness:** Test with intentionally bad models
   (NaN outputs, wrong shapes). The validation step before swap must
   catch these.

---

## What Makes This Different

Most student projects: train model, show accuracy number, done.

This project:
- Multiple "companies" competing in real auctions with real ML
- Model learns from its own competitive environment (not static data)
- Production infrastructure: gRPC, thread pools, circuit breakers
- Observable: every metric on a live Grafana dashboard
- The MDN distribution is what makes multi-strategy bidding possible.
  A point estimate gives every company the same bid. The distribution
  lets each company make a different decision based on risk tolerance.
