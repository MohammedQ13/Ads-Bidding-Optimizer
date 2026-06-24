# Phase 2 Benchmarks

Fresh cloud run: **2026-06-23 on a GCP c2d-standard-16** (16 vCPU, AMD EPYC Milan,
compute-optimized, native Linux). Raw logs in `bench-runs-2026-06-23/`. A budget
e2-standard-16 is included for comparison; the laptop run shows methodology and where
WSL2 caps things.

## First: which role is this? (the framing that makes the numbers make sense)

RTB has two sides. The **exchange / SSP** (Google AdX, Index Exchange, OpenX) is the
*auctioneer*: it runs auctions and fans each one out to many bidders. The
**DSP / bidder** is the *client* it calls, which must return a price inside a hard
deadline or be dropped. **This project is the DSP / bidder** (the C++ fleet). The Go
"auction engine" is a stand-in exchange / visualization harness, not the product. So
the headline metric is a *bidder's*: end-to-end serving throughput at a tight p99
inside the deadline, not a record QPS number.

## Reading the numbers: three layers (read this first)

Several "speed" numbers look contradictory but measure three different things:

1. **Demo market pace: ~200/s paced, ~600/s flat out.** The Go toy-exchange driver
   runs auctions *sequentially* and is paced (`rate_per_sec`, default 200) for a
   watchable demo. **NOT a capacity figure**, it's the auctioneer's demo speed, not
   the bidder's limit. (`RATE_PER_SEC=0` runs it flat out, ~600/s, still sequential.)

2. **Compute ceiling: ~965,000 bids/sec = ~1.04µs/bid.** `bench_engine`, in-process,
   no network: encode + ONNX inference + first-price optimization. The throughput and
   per-bid latency are the same fact (1s / 965k ≈ 1.04µs). This proves the **model is
   never the bottleneck**, the "dyno horsepower" number (engine on a test bench).

3. **End-to-end gRPC serving: ~56k req/s at p99 2.4ms (peak ~79k). THE headline.**
   `bench_async` drives the async server with thousands of requests in flight, the way
   an exchange hammers a DSP. This is the real "bids/sec a node serves over the wire,"
   inside the 10ms deadline at the low-latency operating point.

The one-liner: *the compute kernel does a bid every ~1µs (965k/s); a real node serves
~56k bids/sec end-to-end over gRPC at p99 2.4ms; the live demo market only runs
~200-600/s because the toy exchange is paced/sequential, not because the bidder is
slow.* Millions/sec is **horizontal scale**: stateless nodes, 1 node ~79k to N nodes
~N x 79k. The cloud numbers and full methodology are below.

## Cloud (the real numbers): GCP c2d-standard-16, 16 vCPU, AMD EPYC Milan

Fresh run 2026-06-23, native Linux, compute-optimized cores. Same binary/images as the
demo. Raw logs: `bench-runs-2026-06-23/`.

- **Engine compute: ~965,000 bids/sec = ~1.04µs/bid** (16 threads, batch 256, 30s:
  28,963,328 bids in 30.01s; sweep peak 967,092). Full in-process pipeline, ~60k
  bids/sec per core, no thermal throttling.
- **End-to-end gRPC (async server)**: latency/throughput curve, client + server
  pinned to separate cores:

  | in-flight | req/s | p99 |
  |----------:|------:|----:|
  | light     | 22,000 | 0.54 ms |
  | moderate  | **56,000** | **2.40 ms** |
  | heavy     | 67,000 | 3.98 ms |
  | peak      | ~79,000 | saturated (p99 ~76 ms) |

  Operating point: **~56,000 bids/sec/node at p99 2.4ms, inside the 10ms deadline**;
  push to ~79k req/s if you trade latency. That is ~8-10x the laptop's ~6-8k WSL2 cap,
  which proves that ceiling was the environment, not the design.
- **Budget comparison:** a shared-core e2-standard-16 reproducibly gives ~620k compute
  (same code, slower silicon); c2d's dedicated high-clock cores are what you'd
  actually deploy latency-sensitive inference on.

The bidder is stateless, so a fleet projects linearly (10 nodes ~790k req/s
end-to-end). Cost of the run: a few cents (VM deleted after).

## Laptop (methodology + the WSL2 cap)

All laptop numbers below are on an 8-core Windows machine running Docker Desktop
(WSL2), a noisy, thermally-limited, virtualized environment, so we report the
honest spread, not a cherry-picked figure.

## Headline: core engine throughput

`test/bench_engine.cpp` measures the full bid pipeline in-process (feature
encoding + ONNX inference + first-price bid optimization), with no gRPC, so it
isolates compute capacity. Worker threads run batched inference the same way the
server's micro-batcher does.

- **Per-bid compute floor: ~3 microseconds** (cold cache, batch 64).
- **Burst throughput: ~320,000 bids/sec** on a single node (6 worker threads,
  batch 64; 6 is the sweet spot, 8 oversubscribes the 8 cores).
- **Sustained throughput: ~120,000 bids/sec** after ~10s of continuous load.

The drop from burst to sustained is **thermal throttling**: the CPU runs fast
cold then settles. We confirmed it: six back-to-back 10s runs of the same config
went 212k -> 142k -> 133k -> 124k -> 123k -> 119k, a monotonic settle, not random
noise. On server-grade hardware (no laptop throttling) the cold per-core rate
holds, which projects to **~300k+ bids/sec/node sustained**.

Sweep (clean: server stopped, other local stack paused, all 8 cores):

| threads | batch | bids/sec (burst) |
|--------:|------:|-----------------:|
| 4       | 64    | ~250k |
| 6       | 64    | **~320k (peak)** |
| 8       | 64    | ~190k (oversubscribed) |
| 8       | 256   | ~182k |

Why batching matters: at batch 1 the engine does ~8k bids/sec (dominated by ONNX
per-call overhead); at batch 64 the overhead amortizes and it becomes
compute-bound on the MLP (~33M FLOPs/batch), which is where the 300k+ comes from.

## Async server (the completion-queue upgrade)

The serving path was rewritten from the synchronous gRPC API to the **async
completion-queue API** (`async_server.cpp` + `bid_pipeline.cpp`): a few CQ
polling threads accept RPCs and never block; each request hands off to the
non-blocking pipeline and is finished from the batcher's callback when inference
completes. This decouples in-flight count from thread count.

What it bought us, measured honestly:

- **Graceful high concurrency (the real win).** Under a flood of ~4096 in-flight
  requests, the *sync* server collapsed to ~1.3k req/s and thrashed (a thread per
  in-flight RPC). The *async* server sustains ~8k req/s with **zero errors** and
  no collapse. That is the architectural point of the rewrite.
- **Absolute throughput is environment-bound, not design-bound.** Both servers
  top out around 6-8k req/s on this Docker Desktop / WSL2 setup. We proved it is
  not the threading model and not the batcher: during a flood the batcher sat at
  ~33% utilization with average batch size ~10, i.e. it was *starved*. It is not
  the network either (client+server over localhost was no faster than the bridge).
  The ceiling is the per-request gRPC/protobuf/syscall cost on a nested-virtualized
  WSL2 environment (~100µs+/request of CPU). On bare-metal Linux, gRPC routinely
  does 50-100k+ QPS, so the async server would expose far more of the engine's
  300k/s; the dev environment is the limiter here, not the code.
- Fixes applied along the way (all real): raised HTTP/2 `MAX_CONCURRENT_STREAMS`
  (default 100 was capping concurrency), one client channel per thread (a single
  connection's framing is single-threaded), and pre-created the per-status metric
  counters to drop a locked map lookup from the hot path.

Honest takeaway: the async server is the correct architecture and demonstrably
handles concurrency that breaks the sync server; the raw QPS number is capped by
the laptop's virtualized I/O, while the compute engine (300k/s, no network) shows
what the hardware can actually do.

## End-to-end gRPC latency (sync baseline)

`test/bench.cpp` (closed-loop) and `test/bench_async.cpp` (open-loop, thousands
in flight) drive the real gRPC server. Clean, server on 4 cores, client on 4:

| concurrency | throughput | p50 | p99 |
|------------:|-----------:|----:|----:|
| 16          | ~4.0k req/s| 3.6ms | ~10ms |
| 64          | ~5.5k req/s| 10ms  | ~30ms |
| 128         | ~6.7k req/s| 17ms  | ~45ms |

Honest finding: the **synchronous** gRPC server tops out around 6-7k req/s per
instance and its tail latency climbs under load, because the sync API uses a
thread per in-flight RPC: flooding it with thousands of in-flight requests
(the async client) makes it thrash, not go faster. This is the documented
async-completion-queue upgrade path: the *compute* engine can do 300k+/sec, so
an async server is what is needed to expose that end-to-end. Capping gRPC threads
(`grpc_max_threads`) was tried and made it worse (thread starvation).

At low-to-moderate concurrency the server holds **p99 around 10ms**, inside the
auction deadline budget. Four DSP instances run concurrently in the demo.

## How to reproduce (the 2026-06-23 cloud run)

On a GCP c2d-standard-16 (16 vCPU), same images as the demo:

```
docker build -t cpp-bidder:latest cpp-bidder

# 1) compute ceiling (no network): ~965k bids/sec
docker run --rm cpp-bidder:latest ./build/bench_engine 16 256 30

# 2) end-to-end gRPC, server + client pinned to separate cores
docker network create bn
docker run -d --name bsrv --network bn --cpuset-cpus=0-9 cpp-bidder:latest \
  ./build/bid_server config/config.yaml
# low-latency point (~56k @ p99 2.4ms): threads=4 inflight=16
# peak (~79k, saturated):                threads=6 inflight=512
docker run --rm --network bn --cpuset-cpus=10-15 cpp-bidder:latest \
  ./build/bench_async bsrv:50051 4 16 10

# 3) max-capacity dashboard replay: run the stack flat out (RATE_PER_SEC=0) with
#    bench_async load on the bidders, poll /stats 1/s -> replay_maxcap.json
```

Raw logs from this run: `bench-runs-2026-06-23/{bench_engine,bench_async,bench_async_lowlat}.log`.

## Honesty caveats

- Laptop Docker Desktop: thermal throttling and WSL2 scheduling jitter give a
  real spread. We report burst, sustained, and the methodology.
- Closed-loop p99 is optimistic under saturation (coordinated omission).
- The gRPC ceiling is the sync server, not the engine. Stated plainly above.

## The honest resume framing

Lead with the **end-to-end serving** number, the one that means "a real
DSP bidder", and use the compute number as headroom. Fresh, measured, dated
(2026-06-23, c2d-standard-16):

- "Serves **~56,000 bids/sec on a single node end-to-end over gRPC at p99 = 2.4ms**,
  inside a 10ms auction deadline (~79k req/s peak)."
- "**~965,000 bids/sec of compute headroom (~1µs/bid)** on a 16-vCPU node: the
  model is never the bottleneck; the design optimizes the overhead around it."
- "Stateless and horizontally scalable: millions/sec is a fleet of these nodes
  (1 node ~79k to N nodes ~N x 79k). Four DSP strategies compete live on one shared
  model with a closed retraining loop and zero-downtime ONNX hot-reload."

One-liner: *"Built a horizontally-scalable C++/ONNX demand-side bidder serving ~56k
bids/sec/node end-to-end over gRPC at p99 2.4ms (inside a 10ms deadline; ~79k peak),
with ~965k bids/sec compute headroom (~1µs/bid), the architecture real DSPs use,
scaled by adding stateless replicas."*

If asked for budget/laptop figures, they're honest too: e2-standard-16 (shared cores)
~620k compute; the WSL2 laptop caps at ~6-8k req/s (an environment limit we proved by
hitting 56-79k on real cloud Linux).

Do not claim "100k requests in 10ms" (that is 10M QPS, not real on one node). The
real, defensible story is: a low-latency bidder at ~56k req/s/node and p99 2.4ms,
microsecond compute headroom, horizontal scale, and the distributed feedback loop.
