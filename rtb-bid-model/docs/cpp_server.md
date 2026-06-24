# C++ Inference Server - Design and Scalability

This is the deep design for Phase 2. build_plan.md has the week-by-week
checklist and infrastructure.md has the system-wide overview. This doc is
the reasoning underneath those: why the server is built the way it is, the
tradeoffs behind each choice, and what has to be true for it to scale.

The job is narrow. Take a BidRequest over gRPC, encode features, run one
ONNX forward pass, turn the predicted price distribution into a bid, send
back a BidResponse. The hard part is not the math. The hard part is doing
this with a predictable p99 under hundreds of concurrent clients, on a
model small enough that the serving machinery costs more than the model.


## Design constraints

Four things shape every decision here.

1. The 10ms budget. The whole page-load-to-ad pipeline is under 100ms and
   our slice is roughly 10ms end to end, including the network hop from the
   Go engine. That leaves maybe 2-4ms of actual compute budget inside the
   server. The target is not "fast on average" - it is a tight p99 so the
   Go engine's deadline almost never excludes us from the auction.

2. The model is tiny. The deployed model is a 512-256-128-64 MLP with a
   301-way softmax head (the uniform bins_300 model, where bin k is the
   probability that the clearing price is k fen). A single forward pass is a
   handful of small matmuls, microseconds of real FLOPs. This flips the usual
   serving intuition: inference is not the bottleneck, the fixed overhead
   around it is (tensor setup, kernel dispatch, feature encoding, gRPC
   serialization). Optimize the overhead, not the matmul.

3. Statelessness on the hot path. A bid decision depends only on the
   request and the loaded model. No request needs another request's state.
   That makes the server horizontally scalable by default - the only
   things that break it are budget state (shared across instances) and the
   dedup cache (per instance), both discussed below.

4. Hot-reload from day one. Phase 3 swaps the model live. The server does
   not need the file watcher in Phase 2, but the inference path must read
   the session through an indirection that can be swapped atomically. If we
   hardcode a session pointer now, retrofitting the swap later means
   touching the hot path. Build the seam early, use it later.


## Request lifecycle

Where the time actually goes, in order:

```
gRPC recv + protobuf parse        ~tens of us, arena-allocated
  -> feature encoding             categoricals + quantile transform + tags
  -> (optional) micro-batch wait  up to the flush timeout
  -> ONNX Run()                   one forward pass, shared session
  -> CDF + ternary search bid     ~30 iterations, nanoseconds
  -> protobuf serialize + send    small response, just the bid
```

The two line items worth measuring before optimizing anything are feature
encoding and ONNX call overhead. For this model they are plausibly larger
than the forward pass itself. Everything below follows from taking that
seriously.


## Threading architecture

### gRPC: async, not sync

The sync gRPC API gives you one thread per in-flight RPC. It is simple and
fine at low concurrency. At 500 concurrent clients it means 500 threads
fighting over a handful of cores - context-switch storms, cache thrash,
and a p99 that wanders. We use the async API (completion queues): a small
fixed set of network threads pull completed events off N completion queues
and hand work to a compute pool. In-flight RPC count is decoupled from
thread count, which is the whole point.

Rule of thumb for the completion queues: one or two CQs per core, and the
network threads do nothing but drain them and enqueue work. No inference,
no feature math on a CQ thread.

### Separate the network pool from the compute pool

Two pools, sized independently:

- Network/IO threads: drain completion queues, parse, dispatch. Cheap,
  few of them.
- Compute workers: feature encoding + inference + bid. This is where the
  CPU time lives. Pre-allocated, bounded work queue in front of them.

Keeping them separate means a burst of network events cannot starve
compute and a slow inference cannot block the event loop. The compute pool
is the one that gets pinned and tuned.

### The ONNX Runtime oversubscription trap

This is the single most important threading decision and the easiest to
get wrong. ONNX Runtime has its own intra-op thread pool (parallelism
inside one Run() call) and an inter-op pool (parallelism across nodes). If
we run a compute pool of, say, 2x cores AND let each ORT session spin up
intra_op_num_threads = cores, we get cores x 2x cores threads all wanting
CPU at once. The machine spends its time scheduling threads instead of
doing work, and p99 falls apart.

For a model this small, intra-op parallelism is pure loss anyway - the
matmuls are too small to amortize the cost of forking and joining a thread
team. So:

```
intra_op_num_threads = 1
inter_op_num_threads = 1
parallelism comes from running many single-threaded inferences at once
```

Concurrency lives at the application layer (many compute workers each doing
a sequential forward pass), not inside ORT. The ORT session is thread-safe
for concurrent Run() calls, so all workers share one session and one set of
weights. This is the opposite of what you would do for a large transformer,
and it is correct here precisely because the model is small. Benchmarking
intra_op = 2 confirms it is worse, and the prior is strong.

### Thread affinity

Pin compute workers to cores. Unpinned threads migrate, and every
migration is a cold cache and a tail-latency spike. Pinning costs nothing
and removes a jitter source. On a multi-socket box, keep a worker and the
memory it touches (the model weights, its feature buffers) on the same NUMA
node - a cross-socket memory read is several times slower and shows up
directly in p99.


## Micro-batching

The pitch: one ORT Run() call has fixed overhead regardless of batch size,
so a batch of 32 costs nearly the same as a batch of 1 and gives 32x the
throughput per call. Accumulate requests, flush at batch size N or after
timeout T, whichever comes first.

The honest part: micro-batching is a throughput tool that costs latency,
and for a model this small the payoff is smaller than the textbook case. A
batch of 1 here is already cheap. The real win is amortizing the per-call
overhead (tensor allocation, dispatch) across requests, not amortizing
compute. So batching helps, but it is not free money, and the right N and T
are an empirical question, not a guess.

The tradeoffs that matter:

- T is a latency tax at low load. If requests arrive slowly, each one waits
  up to T for batch-mates that never come. At low QPS the batcher should
  effectively flush immediately (adaptive: if the queue is short, do not
  wait). At high QPS batches fill before T expires and the tax disappears.
- Batching couples unrelated requests. One slow batch delays every member
  (head-of-line). Keep batches small enough that a single flush is well
  inside the SLA budget.
- A dedicated batching thread owns the accumulation buffer and dispatches
  results back through per-request promise/future pairs. The compute worker
  that runs the batch fulfills all the futures at once.

Default to a small T (around 1ms) and a small N (around 32), then sweep
both and report the throughput-vs-p99 Pareto frontier (benchmark 3). The
result might be that batching buys little for this model - if so, that is
a finding worth stating, not a failure.


## Memory and allocation

Latency jitter on a C++ hot path is almost always allocation or page
faults, not compute. The plan is to allocate nothing in steady state.

- Per-worker feature buffers, pre-allocated and reused. A request encodes
  into its worker's existing buffer.
- Reuse input/output tensors via ORT's IoBinding so Run() writes into
  buffers we own instead of allocating fresh ones each call.
- Enable the ORT CPU memory arena so internal allocations come from a
  pre-grown pool rather than the system allocator.
- Arena-allocate protobuf messages (gRPC supports this) so request/response
  parsing does not hit malloc per field.
- Swap in a thread-caching allocator (tcmalloc or jemalloc) so the
  allocations we cannot avoid do not contend on a global lock across
  workers.
- Touch (pre-fault) the model weights at startup, and consider mlock so the
  pages are never reclaimed. A major page fault mid-inference is a
  millisecond-scale stall.

None of this matters for throughput. All of it matters for p99.


## Feature processing is a real cost

It is tempting to treat feature encoding as free glue around the model. For
this system it might be the most expensive step. Per request:

- 9 categorical lookups in pre-reserved hash maps (cheap, but not zero).
- 9 quantile transforms. Each is a binary search on an exported quantile
  array, an interpolation, then an erfinv to map to the normal output that
  the model was trained on. erfinv is not a cheap function.
- Tag vocab lookups for the user tags.

Two things follow. First, the QuantileTransformer must match the Python
sklearn output closely - the arrays are already exported in
feature_config.json, and we validate the C++ inverse transform against
Python on a few thousand samples before trusting it (a silent mismatch here
poisons every bid). Second, if profiling shows erfinv dominating, a
precomputed lookup table with interpolation is an option, trading a tiny
accuracy loss for speed. Do not do that preemptively - measure first.

Cold start: unknown domains, cities, or tags map to a default/unknown
embedding index, and we bump a Prometheus counter so we can see how often
the live traffic falls outside the training vocabulary.


## Hot-reload readiness

Phase 3 retrains and swaps the model live. Phase 2 builds the seam so that
swap is a non-event later.

The session is held behind an indirection the hot path reads on every
request. The simplest correct version is std::atomic<shared_ptr<Session>>:
the swap thread builds a new session, validates it (run a few cached inputs,
check shapes and no NaNs), then stores it; readers load the current pointer,
hold their shared_ptr for the duration of the call, and the old session is
freed when the last in-flight request that referenced it returns. Zero
downtime, zero dropped requests.

The tradeoff to flag now: an atomic shared_ptr load on the hot path is not
free - under heavy concurrency the reference-count traffic can contend.
For our request rates it is almost certainly fine, and we start there
because it is obviously correct. If profiling later shows the atomic load as
a hot spot, the upgrade path is RCU-style or per-worker session pointers
refreshed at a safe point between requests (no per-call refcount at all).
That is a Phase 3 optimization, noted here so the Phase 2 interface does not
foreclose it. The thing Phase 2 must not do is read a raw Session* that
cannot be swapped.


## Backpressure, load shedding, and degradation

When demand exceeds capacity, the server must fail in a way that protects
the requests it can still serve. Three mechanisms, in order of how hard
things are going wrong:

- Bounded work queue. When the compute queue is full, new requests get
  RESOURCE_EXHAUSTED immediately instead of queuing into a latency cliff.
  Shedding a few requests cleanly beats blowing the SLA for all of them.
- SLA-aware dropping. A request carries (or the server stamps) an arrival
  time. If it has already sat past the budget when a worker picks it up,
  there is no point running inference - the Go engine has moved on. Drop
  it or return the fallback and record it.
- Circuit breaker around ONNX. If inference starts failing (corrupt model
  after a bad swap, ORT error), trip CLOSED -> OPEN, serve the fallback bid
  while open, and probe with one request after the cooldown (HALF-OPEN).
  This stops a broken model from turning into a wall of timeouts.

The fallback itself is a precomputed heuristic: historical average clearing
price per ad category times a shading factor (around 0.7). It is a bad bid
compared to the model, but a bad bid that arrives on time still wins some
auctions and keeps the DSP participating. Always bid something. Track the
fallback rate in Prometheus - a rising fallback rate is the early warning
that something upstream is wrong.


## Tail latency checklist

p99 is the product. The known sources of tail latency and what handles each:

- GC pauses - none, this is C++.
- Allocation jitter - pre-allocation + arena + tcmalloc/jemalloc.
- Page faults on weights - pre-fault and mlock at startup.
- Thread migration - pin compute workers to cores.
- Cross-socket memory - NUMA-local weights and buffers.
- Lock contention - shared read-only session, per-worker buffers, no
  shared mutable state on the hot path.
- Logging I/O on the hot path - never log synchronously; use an async
  logger that hands strings to a background thread.
- Head-of-line from batching - keep batch flushes well inside budget.
- Model swap stalls - validate off the hot path, swap atomically.

The discipline is the same throughout: keep the hot path doing arithmetic
on memory it already owns, and push everything else (allocation, I/O,
validation, model loading) off to startup or to background threads.


## Horizontal scaling

Inference is stateless, so scaling out is mostly free: add instances, point
the Go engine at them. Two wrinkles.

First, "company" is not the same as "replica," and the docs have to keep
these straight. A company (A profit-max, B aggressive, C conservative, D
budget-paced) is a distinct configuration: same ONNX model, different
bid_multiplier and strategy. A
replica is a horizontal copy of one company for capacity. So companies are
separate deployments (separate config, separate dsp_id), and each company
can independently run multiple replicas behind a gRPC client-side load
balancer if one instance cannot keep up. For the demo the load is light
enough that one instance per company is plenty; the design just should not
assume a company is a single process.

Second, what breaks statelessness:

- Budget state. Multiple replicas of the same company spend from one
  advertiser budget. Tracked independently, total spend overshoots. This is
  the one piece of genuinely shared state, and it lives in Redis (DECRBY on
  win, checked before bidding) or a small dedicated budget service. It is
  touched only on wins, not every bid, so it stays off the hot path. See
  infrastructure.md for the Redis design.
- Dedup cache. The LRU dedup cache (keyed on request_id, ~500ms TTL) is
  per instance. That is correct as long as a retried request lands on the
  same instance. In this system the Go engine calls each company directly,
  so retries for a given company go to that company - the per-instance
  cache is fine. It would only break under a load balancer that scatters
  retries across replicas, which is worth a note if replicas are added.

HPA signal: scale on a saturation signal, not raw CPU. Request queue depth
and p99 latency track "are we about to miss deadlines" better than CPU
utilization, which can read low while the tail is already bad (or high
during a harmless batch). Queue depth is the cleaner trigger.


## Capacity, back of the envelope

Worth a rough number to sanity-check the architecture before benchmarking.
The forward pass is a few hundred thousand FLOPs - on a modern core that is
well under a microsecond of pure compute. Realistic per-request cost is
dominated by feature encoding (the erfinv-heavy quantile transforms) plus
ORT call overhead, call it low tens of microseconds per request if it is
done carefully. That puts a single core in the rough ballpark of tens of
thousands of bids per second, and a handful of cores comfortably past
100k QPS per instance - far more than a simulated exchange will generate.

The point of that estimate is not the exact number. It is that for this
workload we are overhead-bound and concurrency-bound, not compute-bound,
which is why the design spends its effort on threading, allocation, and
batching overhead rather than on the model. If the real numbers come back
far below this, the profiler tells us which of feature encoding, ORT
overhead, or contention is the culprit.


## Benchmarking methodology

Numbers only mean something if they are measured honestly.

- Warm up first. Run ~10k requests before recording, so ORT JIT and the
  allocator arenas are warm. Cold-start latency is real but it is a
  separate measurement, not part of steady-state p99.
- Report the full distribution: p50, p95, p99, p999, and max. The mean
  hides exactly the tail we care about.
- Mind coordinated omission. ghz and most closed-loop load testers wait for
  a response before sending the next request, which silently hides latency
  during a stall (the load generator slows down with the server). Either
  use an open-loop generator or interpret closed-loop p99 as optimistic and
  say so.
- Find the knee. Sweep concurrency (50/100/200/500/800 clients) and look
  for where latency stops being flat and starts climbing - that knee is
  the real capacity, and it should degrade gracefully (load shedding), not
  collapse.

The four benchmarks from the build plan map onto this directly: thread-pool
scaling (QPS vs pool size), latency vs concurrency (the knee), micro-batcher
tuning (the throughput/p99 frontier), and strategy comparison (a Phase 3
concern, profit per company). Each one should answer a specific question,
not just produce a graph.


## Open questions to settle by measurement

These are deliberately not decided in advance, because the model size makes
the usual answers untrustworthy:

- Does micro-batching actually help for a 301-bin MLP, and at what N/T?
- Is feature encoding (erfinv) or ORT call overhead the bigger per-request
  cost? Which one is worth optimizing?
- Is intra_op = 1 really best, or does intra_op = 2 help at large batch?
- How much does the atomic shared_ptr load cost on the hot path under real
  concurrency - enough to justify the RCU upgrade, or noise?

Each has a default chosen above (batch small, assume overhead-bound,
intra_op = 1, atomic shared_ptr) so development is not blocked, but each
gets revisited once the profiler is pointed at a running server.
