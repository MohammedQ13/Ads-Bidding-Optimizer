# Go Auction Engine (Phase 3)

The exchange. It runs the marketplace where the C++ "companies" compete, and it
doubles as the **telemetry aggregator** for the whole system: it is the one
process that talks to every component, so it collects everything the dashboard
shows and serves it as one rich JSON/SSE feed.

For every auction it makes up an impression, asks each company for a bid (with a
deadline), adds synthetic competitor bids, runs a first-price auction, tells each
company whether it won, and logs the outcome so the retrainer can learn from it.

## What it does, per auction

```
make a random impression (BidRequest)
  -> GetBid from every company at once, with a 10ms deadline (late = excluded)
     (the full BidResponse is captured: bid, P(win), expected profit,
      used_fallback, inference_us, plus the round-trip time the engine measures)
  -> generate synthetic competitor bids (the rest of the market, 8 archetypes)
  -> first-price auction: highest affordable bid wins and pays its own bid
  -> NotifyOutcome to each company (won?, clearing price, profit)
  -> fold it all into the live telemetry + append to /data/outcomes.jsonl
```

## Auction settlement and the profit formula

The auction is **first-price**: the highest affordable bid wins and **pays its own
bid** (`winner.budget -= winningBid`). The clearing price recorded is the winning
bid, or the top competitor bid if that was higher.

Realized profit on a win is:

```
profit = V - bid          // V = impression_value, default 150 fen
```

This is **allowed to go negative** - there is **no `profit < 0` floor** (it was
removed from both the live settle path and the `replayOne` backtest path). A
strategy whose multiplier pushes its bid above `V` (the aggressive 1.2x company bids
~151 > 150) overpays and books a real loss on every win. We surface that loss rather
than hiding it: the scoreboard should show an unprofitable strategy as unprofitable.
Because profit is `V - bid`, the whole scoreboard reduces to `wins x (V - bid)`, so a
lower bidder with a fatter margin per win can lead on profit despite a lower win
rate.

## Synthetic competitor model

Each auction draws **8 synthetic competitor bids** ("the rest of the market"), and a
company must beat the **max of those 8** to win. The bids come from a lognormal
calibrated to the iPinYou clearing prices, scaled by named archetypes (whale 1.6x,
aggressive 1.4x, sniper 1.25x, contender 1.15x, market 1.0x, value 0.85x,
conservative 0.7x, bargain 0.55x). Tuning `competitors` (count, base_median, spread) changes
how hard the market clears, which is the main knob for scoreboard balance. Note that
this max-of-8 synthetic market clears **higher** than the iPinYou distribution the
served model was trained on, and that distribution shift is why the model's predicted
win probability runs above the realized win rate, and it is what the retraining loop
corrects.

## Telemetry it serves (`:9200`)

This replaced the old Prometheus/Grafana plan with a single rich endpoint the
custom console reads:

- `GET /stats`: one JSON snapshot of the whole system
- `GET /events`: SSE stream of that snapshot ~twice a second (CORS open)
- `GET /healthz`: liveness

The snapshot includes, per strategy: economics (won/lost/profit/spend/budget/ROI),
the model's output (avg P(win), expected vs realized profit, avg/last bid,
fallback rate), and latency percentiles for both **round-trip** (measured here)
and **ONNX inference** (reported by the server). System-wide it includes the
clearing/bid/competitor/win-probability histograms, the impression mix
(hour/exchange/slot/floor), the competitor archetype breakdown, a ring buffer of
recent auctions for the live feed, and the retrainer's status.

### Collectors

Two background collectors enrich the feed (toggle with `scrape_metrics`):

- **Prometheus scrape**: every 2s the engine pulls each company's `/metrics`
  page (default `host:9100`, override per company with `metrics_address`) for the
  internals the bid response can't carry: circuit breaker state, batch queue
  depth, average batch size, dedup cache hits, cold-start vocab misses, and the
  loaded model version. It also reads each server's `bid_request_total` counter
  and exposes `served_per_sec` per company, the real serving throughput computed
  as a rate from the counter delta between scrapes. `batch_avg` is a recent
  average too (about 1 at rest, higher under load), computed from the batch
  sum/count deltas rather than the lifetime cumulative average.
- **Retrainer watch**: every 3s it reads the JSON status file the retrainer
  writes (`retrainer_status` in config, default `/models/retrainer_status.json`)
  to surface the training round, window size, last loss, and export count.

## Layout

```
proto/bid_service.proto     copy of the C++ contract (+ a go_package line)
internal/config             YAML config loader
internal/competitors        synthetic "rest of the market" bids (8 archetypes)
internal/stats              the live telemetry store + JSON/SSE server
internal/engine             the auction loop + collectors (scrape.go)
cmd/engine/main.go          startup + shutdown
config.yaml                 companies, auction rate, V, budgets, competitors
```

## Build and run

Everything builds in Docker (the image also generates the Go gRPC stubs from the
proto and runs `go vet` + `go test` as a build gate):

```
docker build -t rtb-engine:latest .
# run it against companies on a docker network (see the root docker-compose.yml)
```

The engine reaches the companies by service name on port 50051 inside the compose
network, and scrapes their metrics on 9100. A `--replay <file>` flag backtests
against a recorded outcome log instead of running live auctions.

## Notes

- Synthetic competitors are drawn from a lognormal calibrated to the iPinYou
  clearing prices, scaled by named archetypes (whale, aggressive, sniper, contender,
  market, value, conservative, bargain). Tuning `competitors` (count, base_median, spread) and the per-company budgets
  changes how balanced the scoreboard is. The default is **8 competitors/auction**.
- Budget pacing: each company has a daily budget that resets every `day_auctions`;
  a company ahead of even-pace sits out some auctions so the others win their share.
  A company can override the shared `auction.daily_budget` with its own `budget:`
  in config. Company-d sets `budget: 6000` (half the shared 12000) so it is
  genuinely budget-paced: it bids 1.0x like company-a but spends its share early
  and paces out, winning about half as many auctions. The mechanism is
  `Company.Budget` + `BudgetFor()` in `internal/config/config.go`, a per-company
  `dailyBudget` seeded in `engine.New()`, and per-company pacing in `runAuction`
  (each company paces against its own daily budget).
- The outcome log is a plain JSONL file (the MVP feedback path). Swapping it for
  Kafka is the documented production version.
