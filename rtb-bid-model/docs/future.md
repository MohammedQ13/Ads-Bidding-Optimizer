# Future Enhancements and Known Limitations

## Known Limitations

### Selection Bias (Winner's Curse)

The training data only contains auctions iPinYou won. Auctions they
lost (where clearing prices were higher than their bid) are missing.
This means our model slightly underestimates the right tail of the
clearing price distribution.

Severity: mild. iPinYou used high bids (~227 fen) for data collection,
well above typical clearing prices (20-200 fen). They won most auctions.
But the bias is nonzero.

Possible fixes for production:
- Use bid logs (all bids including losses) if available
- Censored regression: treat lost auctions as right-censored observations
- Survival analysis: model time-to-event where "event" is clearing at price p
- Importance weighting: upweight samples near the censoring boundary

### No CTR Model

Impression value V is hardcoded at 150 fen. A real DSP computes:

    V = P(click | impression) * value_per_click

This requires a click-through rate model trained on click logs. The
click/conversion logs exist in the iPinYou dataset but were not
downloaded. The Go layer will pass V as a per-auction parameter, so
the ML model does not need V - it only predicts the clearing price
distribution.

### Single Market Period

Training covers 7 days (June 6-12, 2013). Real ad markets shift
constantly. The model needs periodic retraining in production.
The closed-loop simulation in the Go layer solves this: the model
trains on data from its own operating environment continuously.

### No Domain Quality Features

Domain (publisher) names are anonymized hashes. In a real system,
publisher quality is one of the strongest price signals (premium sites
clear much higher). We use domain as a categorical with embeddings,
which captures price associations per domain, but we cannot build
semantic features like "news site vs gaming site."

### First-Price Framing on Second-Price Data

The iPinYou data was collected during the second-price era. We treat
observed clearing prices as competitor bid levels and optimize for
first-price profit. This is a valid approximation because the clearing
price in second-price IS the max competitor bid. But the competitive
dynamics differ: in second-price everyone bids their true value, while
in first-price everyone shades. The distribution we learn reflects
second-price truthful bidding behavior, not first-price shaded behavior.
The closed-loop simulation fixes this: once the Go auction engine runs
first-price auctions against calibrated competitors, the model retrains
on true first-price auction outcomes.

### Temporal Distribution Shift

The train/test split is temporal (June 6-12 train, June 13-15 test).
Market conditions shift between these windows. This is the primary
performance ceiling: the last 5 optimization attempts all landed in
[19.908, 20.001] fen regret, suggesting we have hit the limit of what
offline training on this split can achieve.

## What Was Explored (Phase 1)

These directions were tested during cloud training rounds 1-4:

- **MDN component count**: K=4,6,8,12,20. K=6 was best single,
  K=12 used in config default. Diminishing returns past K=6.
- **DiscreteBins variants**: uniform (301), quantile-spaced (200, 300),
  log-spaced (200, 300), sqrt-spaced (300). Quantile-200 was best.
- **Dropout sweep**: 0.01 to 0.30. Dropout=0.02 was best for MDN,
  0.10 for bins. This was the highest-impact hyperparameter.
- **Architecture variants**: wider (1024-512-256-128), deeper (5 layers),
  narrower (256-128-64). All worse than 512-256-128-64.
- **LightGBM quantile regression**: 14 quantile levels, interpolated CDF.
  21.84 fen, worse than neural bins but ran in minutes.
- **LightGBM regression**: point estimate, 37.13 fen, proved
  distributional prediction is mandatory.
- **Ensemble search**: exhaustive weight search over model combinations.
  Best: 3 MDN + 2 bins at 19.908 fen.
- **Entropy regularization**: encourages MDN to use all K components.
  Marginal improvement.
- **Target jitter**: noise on log(payprice) during training. Marginal.
- **Label smoothing**: Gaussian-smoothed CE for bins. Helped calibration
  but not regret.
- **Per-auction V (bidding_price)**: tested as impression value. Newton
  optimizer was unstable; grid search worked but didn't improve regret
  meaningfully since bidding_price is linearly scaled.
- **Newton bid optimizer**: unstable on multimodal CDFs, abandoned in
  favor of grid search.

## Remaining ML Improvements

### Uncertainty Calibration

After training, the PIT (Probability Integral Transform) check tells us
if the predicted distributions are calibrated. If the PIT histogram is
not uniform, apply post-hoc calibration (isotonic regression or Platt
scaling on the CDF values). Tested during Round 1 - improved NLL but
hurt regret. May be worth revisiting with bins models.

### Per-Advertiser Conditioning

The 5 advertisers have different price dynamics. The model uses
advertiser_id as a categorical embedding, which partially handles this.
Could also try per-advertiser heads or separate fine-tuned models.

### Feature Interactions

The MLP learns interactions through hidden layers. Could try explicit
interaction features:
- ad_exchange x slot_visibility (auction dynamics differ by exchange)
- hour x is_weekend (weekend traffic patterns differ)
- advertiser_id x slot_format (different formats per campaign)

Risk: increased input dimensionality, possible overfitting.

### Online Learning (Sliding Window)

With the Go feedback loop:
- Sliding window of last W simulation outcomes (e.g. 500K)
- Fine-tune current checkpoint on the window periodically
- Oldest data drops out as new data arrives, which prevents synthetic
  accumulation and keeps the model grounded in recent conditions
- iPinYou-trained checkpoint is the warm start (real auction priors)
- Pointer swap in C++ servers: background thread loads new ONNX,
  atomic shared_ptr swap, zero downtime
- Drift detection: monitor val NLL on window, auto-retrain if degrading

### ONNX Export (done)

The deployment model is exported. It is the uniform bins_300 (301 bins,
20.33 fen), verified against the PyTorch outputs (max diff 4.5e-7). That
file (exports/best_model.onnx) is what the C++ server loads in Phase 2.

bins_quant200 (0.16 fen better) was not deployed. Its bins are quantile-
spaced, so the bin index is not the price in fen; it maps through an edges
array that was never exported. The C++ pipeline assumes bin index equals
price, which only holds for uniform bins, so deploying quant200 as-is gave
wrong bids (regret around 30). Uniform bins_300 makes that assumption
correct with no code change, at a cost of 0.16 fen.

The ensemble (19.908 fen) was not chosen for deployment either. See the
Model Deployment Decision section in build_plan.md for the full rationale.
The single model is simpler to serve and the ensemble advantage disappears
once the feedback loop retrains on live data.

## Planned Infrastructure Improvements

### Budget Pacing

Real DSPs spread budget evenly over the day. The Go layer should:
- Track spend rate per advertiser
- Adjust bid multiplier based on remaining budget and time
- Implement smooth pacing (gradual adjustment) and throttled pacing
  (skip auctions when overspending)

### Auction Replay

Save all auction requests and outcomes to a log. Replay with different
model versions or strategies to compare performance offline. Cheaper
than running live A/B tests.

### Real-time Dashboard (BUILT, custom console, not Grafana)

We dropped Grafana for a custom Next.js telemetry console (`dashboard/`) fed by
the Go engine's JSON/SSE API. Every panel originally planned for Grafana now
exists in it, plus more (live topology, the model's bid vs P(win) curve, calibration,
the live auction stream, the competitor-archetype market mix):

- QPS / throughput across all DSP instances: done (KPI strip + topology)
- Latency percentiles p50/p95/p99: done (round-trip and ONNX inference, vs the
  bid deadline, in the Latency budget panel)
- Win rate + profit per won auction: done (strategy cards + profit race)
- Model version per instance: done (scraped from each `/metrics`)
- Circuit breaker state, fallback rate: done (C++ internals table + cards)
- Budget utilization per advertiser: done (per-strategy budget bars)

See `dashboard/README.md`. The engine scrapes each C++ `/metrics` and reads the
retrainer status file, so the console is the single pane for the whole system.

## Technical Debt

- Feedback pipeline: the MVP streams auction outcomes through a JSONL file.
  Production should use a message queue (Kafka or similar) to stream
  results back for retraining.
- Model versioning: needs a proper registry, not just file paths. Track
  which version runs on which instance, with rollback capability.
- Bid computation in C++: the optimizer scans every price bin (301 of them)
  per request. That is already cheap, a few hundred multiply-adds, but a
  smarter search such as ternary search could reduce it further. The catch
  is that expected profit is not guaranteed unimodal in the bid for an
  arbitrary predicted CDF, which is the same reason the Newton optimizer was
  abandoned in favour of the full scan, so any such change needs care.
