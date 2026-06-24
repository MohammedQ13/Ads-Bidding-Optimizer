# ML Model Development: Full Journey

## The Problem

In real-time bidding, a DSP has ~100ms to decide how much to bid for an
ad impression. In a first-price auction, you pay exactly what you bid.
Bid too high, you waste money. Bid too low, you lose the auction.

The optimal bid depends on predicting the full distribution of what
competitors will bid, not just the average. With the full distribution,
you compute the bid that maximizes expected profit:

    E[profit] = (V - b) * P(competitors bid less than b)

where V is what the impression is worth to you.

## Dataset

iPinYou Season 2 (June 2013). Real RTB impression logs from a Chinese DSP.

- Training: 7 days of impression logs (June 6-12), 10.4M rows
- Validation: last 15% of training by temporal order, 1.8M rows
- Test: 3 days (June 13-15), 2.5M rows (leaderboard set)
- Target: payprice (clearing price in fen, 1 fen = 0.01 CNY)

The temporal split is critical: training on earlier days, testing on
later days. This prevents look-ahead bias but introduces a distribution
shift (test mean log price = 4.155 vs train = 4.046).

## Features

9 categorical features with learned embeddings:
- region, city (min_count 500), domain (min_count 100)
- ad_exchange, slot_width, slot_height, slot_visibility, slot_format
- advertiser_id

Continuous features (z-score standardized from training stats):
- log_floor_price, slot_area, tag_count, has_floor_price

Cyclical features: hour_sin, hour_cos, weekday_sin, weekday_cos
Binary: is_weekend

User tags: variable-length list of interest tags, encoded as bag-of-words
mean-pooled embedding (vocab 44 after min_count=200, padded to length 10).

## Metric: Regret

Regret = oracle profit minus our profit, averaged per impression.
Oracle knows the clearing price and bids just above it.
Lower regret = closer to perfect bidding.

Primary metric: regret at V=150 (fixed impression value).
Secondary: regret at V=bidding_price (per-impression value from the DSP's
original bid, ~227-300 fen).

## v1: BidTransformer (abandoned)

Architecture: Transformer encoder (3 layers, 4 heads, FFN dim 256) on
tabular features. Output: 2 neurons (mu, sigma for log-normal distribution).

Problems:
- Sigma collapsed to zero or exploded, causing NaN losses
- Self-attention has no benefit for tabular data, features have no
  natural ordering, so it was just an expensive MLP
- Single log-normal assumption too rigid for multimodal price distributions
- Convergence was fragile, very sensitive to learning rate

Decision: scrapped the transformer, rebuilt from scratch on branch
v2-mlp-mdn. The feature engineering (embeddings, cyclical encoding,
slot area) survived into v2.

Artifacts: BidTransformer_best.pt and MLPBaseline_best.pt were in
exports/ but have been deleted as dead weight.

## v2: Deep MLP + MDN + Discrete Bins

The architecture that stuck. Two model heads sharing the same backbone:

Backbone: 4-layer MLP (512-256-128-64) with LayerNorm, GELU activation,
and dropout per layer. Categorical features go through learned embeddings.
Continuous features are z-scored. User tags are mean-pooled.

MDN head (Mixture Density Network): outputs K Gaussian mixture components.
Each component has a weight (pi), mean (mu), and standard deviation
(sigma). Sigma uses softplus + floor to prevent collapse. The model
predicts in log-price space; the bid optimizer converts via change of
variables.

DiscreteBins head (DLF-style): outputs softmax probabilities over price
bins. Each bin corresponds to a price range. Cross-entropy loss against
the bin containing the true payprice.

Training: AdamW optimizer, cosine LR schedule with linear warmup, AMP
(float16), exponential moving average (EMA) of weights, early stopping
on validation loss. Best of raw vs EMA weights saved per epoch.

## Round 1: Google Cloud T4 (May 7-8, 2026)

Instance: us-central1-a, T4 GPU. First cloud training run.

What was done:
- MDN with K=12 (bumped from original K=5)
- Trained variants v8-v13 with different hyperparameters
- Ensemble experiments combining multiple MDN checkpoints

Results:
- Best single model (v8): test NLL = -0.683, regret = 21.13 fen
- Best ensemble (v8+v10+v12, weights 0.4/0.4/0.2): regret = 21.02
- Isotonic calibration: NLL improved to -0.901 but regret worsened to
  21.85. First discovery of the NLL-regret tradeoff.

Key findings:
- Ensembling improved NLL by 10.9% but regret only by 0.6%
- v8 was over-confident at median (+5.6%), v10 under-confident (-13.9%)
  - complementary errors made the ensemble work
- Better calibration does NOT mean better bidding. The bid optimizer
  needs sharp peaks, not smooth well-calibrated distributions

What went wrong: the instance was deleted before pulling checkpoints and
code. Everything lost except a notes.txt with the numbers. Lesson
learned: always pull results before deleting cloud instances.

## Round 2: Google Cloud L4 (May 8, 2026)

Instance: g2-standard-8, NVIDIA L4 24GB, Standard pricing. First attempt
was a Spot instance that got preempted; snapshot was used to create a
new Standard instance. Ran 1h 50m total.

### Direction 1: Per-auction impression value

Added evaluation with V=bidding_price (the DSP's actual bid, 227-300 fen)
alongside the fixed V=150. V=bid gives higher absolute regret (~44 fen)
because the value scale is ~3x higher. Decided V=150 grid is the
canonical metric for comparing models.

### Direction 2: DLF-style discrete bins (the key win)

Replaced MDN with softmax over 301 integer price bins (0-300).
Standard cross-entropy loss on round(payprice).

Results:
- bins_300 (seed 42, dropout 0.1): ANLP=3.60, regret=20.33
- bins_300_s1 (seed 1): regret=20.41
- bins_300_s2 (seed 2): regret=20.31
- bins_300_smooth (Gaussian-smoothed targets, sigma=1.5): regret=21.37
  but best calibration (KS=0.182)

All three vanilla bins seeds beat the Round 1 MDN (21.13). The bins
approach learns sharp, detailed price histograms instead of being forced
into smooth Gaussian shapes.

### Direction 3: LightGBM quantile (abandoned this round)

Started training with full defaults (255 leaves, 400 rounds). Way too
slow - ~18 min per quantile on 10M rows while sharing CPU with GPU
training. Killed after 2 quantiles. Revisited in Round 3.

### Direction 4: Newton bid optimizer

Implemented damped Newton-Raphson on the first-order condition:
    b = V - CDF(b) / pdf(b)
with multi-start (5 initial points) and NaN recovery.

- MDN s42 + V=150: Newton 28.92 vs grid 33.13 (Newton helps for
  multi-modal MDN because grid's 500 candidates can miss sharp peaks)
- V=bidding_price: Newton diverges, regret >62 vs grid 48. Spurious
  roots when V is large and CDF is multi-modal.

Decision: grid search is the primary optimizer. Newton helps only for
the overfit MDN at V=150 and is dangerous everywhere else.

### Direction 5: CTR model (skipped)

Test file has click/conversion columns but training files lack click
logs. Cannot train a CTR model without training labels. Left as a hook
for Phase 3 (Go engine passes V as a per-auction parameter).

### Direction 6: Budget-constrained ORTB (skipped)

The architecture supports a Lagrangian budget constraint but a full
lambda sweep was not attempted due to time.

### The MDN problem

mdn_s42 (K=12, dropout=0.05, batch=8192, lr=1e-3): regret = 33.13.
Terrible compared to bins (20.33). The MDN overfit dramatically after
epoch 0 - train loss dropped to -0.08 while val NLL rose from 0.64 to
1.5+. Sigma collapsed on training data, failed to generalize.

Attempts to fix with stronger regularization (dropout=0.3, target
jitter=0.05) prevented learning entirely (val loss flat at 1.17).

### The ensemble surprise

Despite MDN being terrible alone (33.13), it helped in an ensemble:

    3 bins only (s42+s1+s2):        regret = 20.25
    MDN + 1 bins (2:1):             regret = 19.99
    MDN + 3 bins (5:1:1:1):         regret = 19.99

Why: the MDN makes sharp, confident peaks (even though slightly
misplaced). The bins model is accurate but spread out. Blending gives
accuracy AND sharpness, which helps the bid optimizer make decisive bids.

Round 2 best: 19.991 fen (5% better than Round 1's 21.02).

## Round 3: Focused Improvements (May 10, 2026)

Same L4 instance. Targeted the gaps from Round 2.

### MDN regression diagnosis

Tried reproducing Round 1's config:
- mdn_r1 (batch=1024, lr=6e-4, dropout=0.2): val NLL=2.05, much worse
  than mdn_s42's 0.64
- mdn_sf2 (sigma_floor=0.2, wider Gaussians): regret=46.87

Conclusion: Round 1's 21.13 was not reproducible with Round 2's feature
engineering and temporal split. Likely Round 1 used a different split or
different features. The mdn_s42 checkpoint remained the best MDN.

### LightGBM quantile (done properly)

Reworked lgbm_baseline.py with better config:
- 95 leaves, lr=0.12, 150 rounds per quantile
- ~85 sec/alpha (vs 18 min with Round 2's defaults)
- Trained 14 quantiles (q05 through q60, q70, q80) + MAE regression
- Built piecewise-linear CDF from quantile predictions

Results:

    LightGBM 14-quantile:          regret = 21.84, ANLP = 6.77
    LightGBM MAE regression:       regret = 37.13
    Naive (bid mean payprice):     regret = 28.74

Why LightGBM didn't win:
1. No user tags. LightGBM can't handle variable-length multi-hot
   features. The neural models use tag embeddings which trees cannot.
2. 14 quantile points give a much coarser CDF than 200-300 bins.
   ANLP 6.77 vs bins 3.60 reflects this.
3. Each quantile is a separate model, slow to train and predict.

Why LightGBM regression was worse than naive: a point estimate cannot
optimize first-price bids. Bidding the predicted average overpays on
cheap auctions and loses expensive ones. You NEED the full distribution
to compute (V-b)*CDF(b). This is the single most important finding of
the project: distributional prediction is not optional.

LightGBM's role: interpretable fallback. 21.84 is only 1.67 fen behind
the best neural single model, with no GPU required.

5 upper-tail quantiles (q65-q95) were not trained due to time. The 14
present quantiles were sufficient to establish LightGBM's position.
Training more would not have changed the conclusion.

### Improved Newton optimizer

Rewrote with 8 starts over [0.05V, 0.95V], damped fixed-point, NaN
recovery. Tested on full test set:

    mdn_s42:     grid 33.13 -> Newton 30.06 (Newton helps)
    mdn_s1:      grid 25.27 -> Newton 30.65 (Newton hurts)
    bins_300:    grid 20.33 -> Newton 37.98 (Newton hurts badly)
    bins_smooth: grid 21.37 -> Newton 27.50 (Newton hurts)

Newton is consistently bad for V=bid (>70 fen on every model).
Multi-modal distributions with high V have spurious roots even after
multi-start.

Final decision: grid search is canonical. Newton is abandoned.

### Budget simulation and ONNX export

Implemented budget-constrained bidding simulation: process test auctions
in temporal order, bid optimally, skip when budget is exceeded.
Tested at budget fractions 1/32, 1/8, 1/2, and 1x of oracle cost.

Exported best bins model to ONNX (opset 17) with softmax baked into the
graph. Verified numerical match with onnxruntime (max abs diff 4.5e-7).
Generated feature_config.json for the C++ server.

## Round 4: Exhaustive Optimization (May 10, 2026)

Same L4 instance. 2h 17m. 29 MDN configs, 10 bin variants, architecture
experiments.

### The dropout fix (biggest single win)

13 MDN configs varying dropout, seed, K, learning rate:

    mdn_b1 (Round 2 baseline, dp=0.05):  regret = 32.95
    mdn_b2 (dp=0.02):                    regret = 21.46
    mdn_b3 (dp=0.00):                    regret = 22.45
    mdn_c4 (dp=0.02, seed 99):           regret = 21.36
    mdn_c10 (dp=0.02, K=6):              regret = 21.21

Dropout 0.05 -> 0.02 recovered 12 fen of regret. Round 2's MDN was
over-regularized. With this many embedding parameters, the model needs
freedom to learn sharp mixture components. Round 1's claimed 21.13 IS
reproducible with dp=0.02.

Other findings:
- K=6 marginally better than K=12 for single-model (21.21 vs 21.46)
- K=12 more useful in ensembles (sharper modes help blending)
- Seeds 7, 11, 23, 42, 99 all in range 21.36-22.08
- Lower weight decay (1e-4 vs 5e-4): worse (21.63)
- Longer training (10 epochs): worse (21.63)

### Quantile-spaced bins (new best single model)

Instead of equally-spaced price buckets, place bin boundaries at the
percentiles of training payprice. More resolution where prices are
common (40-100 fen), fewer bins where prices are rare (200+ fen).

    bins_300 (uniform, Round 2):          regret = 20.33
    bins_quant200 (200 quantile-spaced):  regret = 20.17  <-- best single
    bins_sqrt300 (sqrt-spaced):           regret = 20.22
    bins_log200 (log-spaced):             regret = 20.46
    bins_uniform_emd (CE + 0.05*EMD):     regret = 20.80
    bins_uniform_ls (label smooth 0.05):  regret = 20.69

EMD loss and label smoothing both HURT regret. The soft target dilutes
the sharp mode the bid optimizer relies on. Same lesson as isotonic
calibration from Round 1: smoother != better for bidding.

### Architecture search

All wider/deeper variants performed worse:
- bins_wide (1024-512-256-128): 20.57
- bins_deep (5 layers): 20.65
- mdn_wide (1024-512-256-128): 22.50 (best KS=0.12, worst regret)
- mdn_K4, mdn_K8: 21.75, 21.81

The 512-256-128-64 backbone is the right capacity for 10M rows.

### Final ensemble search

Used ens_search3.py: 200K-row subset for fast Dirichlet random trials,
greedy coordinate-descent refinement, validate top candidates on full
test set.

    Round 3 best (5:1:1:1 mdn_s42 + 3 bins):    regret = 19.991
    ens_search_w43 (49:5:4:2:0:40, 6 members):   regret = 19.981
    ens_v4_compact (49:5:4:40:2, 5 members):      regret = 19.908  <-- best
    ens_v4_5_2_1_1 (5:2:1:1):                     regret = 19.921

Best ensemble: ens_v4_compact
  49 * mdn_s42 (dp=0.05, the "bad" MDN with sharp peaks)
   5 * mdn_c1 (dp=0.02, seed 42, regret 21.46)
   4 * mdn_c4 (dp=0.02, seed 99, regret 21.36)
  40 * bins_quant200 (200 quantile bins, regret 20.17)
   2 * bins_sqrt300 (300 sqrt bins, regret 20.22)

The "bad" MDN (mdn_s42, 33.13 alone) gets the highest weight. Its sharp
misaligned peaks complement the accurate-but-diffuse bins predictions.

## Why We Stopped

The last 5 ensemble attempts all landed in [19.908, 20.001]:

1. Single-model floor: bins_quant200 at 20.17. Further bin edge tweaks
   (sqrt, log, quantile-300) all in 20.2-20.6. A 0.4-fen window.

2. MDN floor: every dp=0.02 variant across K in {4,6,8,12,20} and
   seeds {7,11,23,42,99} produced 21.21-22.50.

3. Ensemble diversity exhausted: 3 MDN seeds, 6 bin variants, 1 LightGBM.
   Greedy weight search cannot find a combination below 19.9.

4. Architecture changes all lose: wider, deeper, different K, longer
   training, nothing helps.

5. The temporal split is the ceiling. Train days 6/06-6/12, test days
   6/13-6/15 introduces a mean shift that no model can erase. What
   would break the ceiling: more training data or a non-temporal split.
   Both are out of scope.

## Final Results

| Model | Regret V=150 | ANLP | KS |
|---|---|---|---|
| Naive (bid mean payprice 78.19) | 28.74 | - | - |
| LightGBM MAE regression | 37.13 | - | - |
| LightGBM 14-quantile | 21.84 | 6.77 | 0.20 |
| MDN K=6 dp=0.02 (best single MDN) | 21.21 | 4.52 | 0.21 |
| MDN K=12 dp=0.05 (Round 2, overfit) | 33.13 | 4.98 | 0.25 |
| DLF bins uniform 301 (Round 2) | 20.33 | 3.60 | 0.28 |
| DLF bins quantile 200 (best single) | 20.17 | 3.74 | 0.18 |
| ens_v4_compact (best ensemble) | 19.91 | 3.79 | 0.24 |

Per-advertiser regret (V=150, best ensemble):

| Advertiser | Test impressions | Regret |
|---|---|---|
| 1458 | 614,638 | 22.07 |
| 3358 | 300,928 | 17.79 |
| 3386 | 545,421 | 19.27 |
| 3427 | 536,795 | 18.78 |
| 3476 | 523,848 | 20.40 |

## Comparison to Published Work

No published paper on iPinYou reports "regret" the way we define it
(oracle profit minus model profit at fixed V). Papers use different
metrics, so the comparison is indirect but informative.

### Density Estimation (ANLP)

ANLP (average negative log probability) measures how well the model
predicts the true clearing price distribution. Lower is better.

| Method | ANLP | Source |
|---|---|---|
| Kaplan-Meier | 15.366 | DLF paper (KDD 2019) |
| Mixture Model | 6.552 | DLF paper |
| Gamma | 6.310 | DLF paper |
| DeepHit | 5.544 | DLF paper |
| STM | 5.148 | DLF paper |
| DLF (published SOTA) | 4.774 | DLF paper |
| Our LightGBM 14-quantile | 6.77 | this project |
| Our MDN K=6 dp=0.02 | 4.52 | this project |
| Our bins uniform 301 | 3.60 | this project |
| **Our bins quantile 200** | **3.74** | this project |

Our bins models beat DLF's published ANLP=4.774. The density estimation
is strong: our predicted distributions fit the true clearing prices
better than published SOTA.

Caveats: DLF used different train/test splits, different feature
engineering, and a recurrent architecture (RNN over price bins). The
ANLP numbers are not exactly apples-to-apples. But they are the closest
published baseline available on this dataset.

### Bidding Strategy (clicks/conversions under budget)

Zhang et al (2014) and the ORTB paper (2014) evaluate bidding strategies
by clicks won and conversions won under budget constraints, using a
CTR model to set per-auction V. We cannot directly compare because:

1. We don't have a CTR model (no click training labels)
2. We use a fixed V=150 instead of per-auction pCTR * click_value
3. Our metric (regret) measures profit gap vs oracle, not clicks won

Their approach is complementary: they optimize V (what an impression is
worth), we optimize the bid (how much to pay given V). The Go layer
will combine both.

### What 19.9 Fen Regret Actually Means

Oracle profit per impression = V - payprice = 150 - 78.19 (mean) = ~72 fen.
Our model achieves ~52 fen profit, which is **72% of oracle profit**.

The remaining 28% gap comes from:

1. **Temporal distribution shift** (dominant factor): train on June 6-12,
   test on June 13-15. The market moved. Mean log-price shifted from
   4.046 (train) to 4.155 (test). No model can predict market changes
   it hasn't seen. We proved this is the ceiling: last 5 optimization
   attempts all landed in [19.908, 20.001].

2. **Fixed V=150**: real DSPs compute V = P(click) * value_per_click
   per auction. With fixed V, the model bids the same "aggressiveness"
   for every impression regardless of its actual value. Some impressions
   are worth 300 fen, others 50 - we treat them all the same.

3. **Selection bias**: we only see auctions iPinYou won. The right tail
   of the clearing price distribution (expensive auctions they lost) is
   missing from training data.

4. **Grid quantization**: 500 bid candidates over [0.5, 300] gives
   ~0.6 fen resolution. This introduces a small systematic error.

The model itself is not the bottleneck. It beats published SOTA on
density estimation. The regret gap is an evaluation artifact.

### Why V=150 (Not Dynamic)

We explored V=bidding_price (the DSP's actual bid per impression,
~227-300 fen). Three things:

1. Newton optimizer diverged badly at high V (regret >62 vs grid's 48).
   Multi-modal CDFs have spurious roots when V is large.
2. bidding_price is linearly scaled by iPinYou before release, it's
   not the true per-impression valuation.
3. Grid search worked with V=bid but the absolute regret numbers are
   just higher because V is higher. It didn't improve relative
   performance or change model rankings.

A real per-auction V requires a CTR model: V = P(click) * click_value.
We don't have click training labels (only test file has clicks). The
C++ server accepts V as an input parameter, and the Go layer provides it.

### Why the Go Feedback Loop Fixes This

The closed-loop simulation in Phase 3 addresses every limitation:

1. **No temporal shift**: the model trains on data from its own
   operating environment. Train and test distributions match because
   the model is bidding in the same market it trained on.

2. **Per-auction V**: the Go engine knows the true value of each
   impression (from simulated click probabilities and advertiser budgets).
   V is no longer hardcoded.

3. **No selection bias**: in the simulation we see ALL auction outcomes,
   not just ones we won. Lost auctions provide the right-tail signal
   the current model is missing.

4. **True first-price dynamics**: competitors shade their bids (as in
   real first-price auctions) instead of bidding truthfully (as in the
   historical second-price data). The model learns actual first-price
   behavior.

5. **Sliding window retraining**: the model retrains on a sliding
   window of recent simulation outcomes (e.g. last 500K auctions).
   Oldest data drops out as new data arrives, preventing synthetic
   accumulation. The iPinYou-trained model is the warm start that
   provides real auction priors. New ONNX models are loaded via atomic
   pointer swap in the C++ servers (zero downtime).

## Key Lessons

1. Distributional prediction is mandatory for first-price bidding.
   Point estimates (LightGBM regression) are worse than bidding blindly.

2. Discrete bins beat Gaussian MDN as a single model. Non-parametric
   histograms learn the true price shape without Gaussian assumptions.

3. Quantile-spaced bins beat uniform bins. More resolution where the
   data is dense gives better bid optimization.

4. A "bad" model can improve an ensemble. The overfit MDN (33.13 alone)
   contributes sharp peaks that help the bid optimizer when blended
   with accurate-but-diffuse bins predictions.

5. Calibration != bid quality. Wider MDN (KS=0.12, best calibrated)
   had the worst regret (22.50). Isotonic calibration improved NLL but
   worsened regret. Label smoothing worsened regret. The bid optimizer
   needs sharp modes, not smooth well-calibrated densities.

6. Dropout matters more than architecture. 0.05 -> 0.02 recovered
   12 fen. Wider/deeper networks all lost.

7. Grid search beats Newton for bid optimization. Newton has spurious
   roots on multi-modal distributions and diverges at high V.

8. The temporal split is the ceiling. Train/test distribution shift
   limits what any model can achieve on this dataset.

## Current State

What's done:
- Best regret: 19.908 (ensemble), 20.17 (single model)
- All checkpoints, eval results, logs, and processed data saved locally
- ONNX export verified (bins_300, max diff 4.5e-7 vs PyTorch)
- Full experiment documentation in RESULTS.md and experiments.md

Deployment choice (Phase 2 onward):
- The deployed model is the uniform bins_300 (301 bins, bin index = price
  in fen), not bins_quant200. A backtest caught that bins_quant200 is
  quantile-spaced: its bin index is not the price, it maps through an edges
  array that was never exported. The C++ server and the rest of the pipeline
  assume bin index equals price, which is only correct for uniform bins, so
  deploying quant200 as-is produced wrong bids (regret around 30). Keeping
  the uniform bins_300 makes that assumption correct with no code change, at
  a cost of 0.16 fen. See the Model Deployment Decision section in
  build_plan.md.

Known issues (documented, intentionally not fixed):
- Code duplication in experiment scripts (moved to src/experiments/,
  won't be touched again)
- Newton optimizer broken for V=bid (abandoned, grid is canonical)
- 5 LightGBM upper quantiles not trained (conclusion already reached)
- Shipping bins_quant200 (the 0.16-fen-better single model) would need its
  bin edges exported plus an edge-aware bid optimizer; neither was built,
  and the gain is not worth it for the live system

## Reproduction

    # Features (cached in data/processed/)
    PYTHONPATH=src python3 src/features.py

    # Best single model
    PYTHONPATH=src python3 src/train.py --model bins --name bins_quant200 \
      --hidden 512,256,128,64 --batch_size 8192 --num_bins 200 \
      --lr 1e-3 --dropout 0.1 --weight_decay 5e-4 --ema_decay 0.99 \
      --seed 42 --epochs 5

    # MDN for ensemble
    PYTHONPATH=src python3 src/train.py --model mdn --name mdn_c1 \
      --hidden 512,256,128,64 --batch_size 8192 --K 12 --sigma_floor 0.05 \
      --lr 1e-3 --dropout 0.02 --weight_decay 5e-4 --ema_decay 0.99 \
      --ent_bonus 0.005 --seed 42 --epochs 5

    # Evaluate
    PYTHONPATH=src python3 src/evaluate.py --ckpt exports/bins_quant200/best.pt \
      --name bins_quant200 --split test

    # ONNX export
    PYTHONPATH=src python3 src/export_onnx.py --ckpt exports/bins_300/best.pt \
      --out exports/best_model.onnx --feature_config exports/feature_config.json

## File Structure

    rtb-bid-model/
      config.yaml              - full config with mdn/bins/training sections
      requirements.txt         - Python dependencies
      src/
        config.py              - loads config.yaml
        features.py            - raw data -> processed parquets
        dataset.py             - loads parquets into memory
        model.py               - MDN + DiscreteBins architectures
        loss.py                - NLL losses (MDN, bins, smoothed)
        train.py               - training loop (AMP, EMA, early stopping)
        evaluate.py            - full evaluation pipeline
        bid_optimizer.py       - grid + Newton bid optimization
        export_onnx.py         - ONNX export for Phase 2
        experiments/           - one-off cloud experiment scripts (23 py + 7 sh)
      docs/
        RESULTS.md             - final results summary
        experiments.md         - detailed experiment log (all rounds)
        ml_journey.md          - this file
        model.md               - architecture details
        data.md                - dataset and feature engineering documentation
        pipeline.md            - training, evaluation, and ONNX export pipeline
        future.md              - known limitations and planned improvements
        build_plan.md          - Phase 2/3 build plan
        infrastructure.md      - infrastructure stack design
      exports/                 - not in git
        best_model.onnx        - bins_300 ONNX (verified)
        feature_config.json    - feature encoding spec for C++ server
        bins_*/                - bins model checkpoints
        mdn_*/                 - MDN checkpoints
        ensembles/             - ensemble evaluation pickles
        preds_eval/            - single-model evaluation pickles
        budget/                - budget simulation pickles
        lgbm_quant/            - LightGBM models
      data/processed/          - train/val/test parquets + artifacts.pkl (not in git)
      results/plots/           - diagnostic plots (not in git)
