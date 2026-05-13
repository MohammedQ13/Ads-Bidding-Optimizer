# RTB Bid Engine

Real-time bidding engine for first-price ad auctions. Predicts the full
distribution of competitor bids and picks the bid that maximizes expected
profit.

This is the ML layer (Layer 1) of a three-layer system:
1. Python ML model (this repo) -- trains on iPinYou data, exports ONNX
2. C++ inference server -- loads ONNX, serves bids over gRPC
3. Go auction engine -- runs simulated auctions with synthetic competitors

---

## How It Works

In a first-price auction you pay what you bid. Bid too high, waste money.
Bid too low, lose the impression. So the key is predicting what others
will bid.

The model takes auction features (ad slot size, website, user tags,
time of day, etc.) and outputs a probability distribution over clearing
prices. The bid optimizer then picks the bid maximizing:

    E[profit] = (V - bid) * P(competitors bid less than bid)

where V is how much the impression is worth to the advertiser.

Why not just predict the average price? Because that loses. We tested it:
LightGBM regression (point estimate) got 37.13 fen regret, worse than
just bidding the same number every time (28.74). You need the full
distribution to bid well.

---

## Results

Best ensemble regret: **19.908 fen** per impression (vs oracle).

| Model | Regret V=150 | ANLP | KS |
|---|---|---|---|
| Naive (bid mean payprice) | 28.74 | -- | -- |
| LightGBM regression (point estimate) | 37.13 | -- | -- |
| LightGBM 14-quantile | 21.84 | 6.77 | 0.20 |
| Best MDN single (K=6, dp=0.02) | 21.21 | 4.52 | 0.21 |
| Best bins single (200 quantile-spaced) | 20.17 | 3.74 | 0.18 |
| **Best ensemble (3 MDN + 2 bins)** | **19.91** | 3.79 | 0.24 |

Regret = profit left on the table vs a perfect oracle. Lower is better.
Our best density estimation (ANLP 3.74) beats the published SOTA on
iPinYou -- DLF from KDD 2019 reported ANLP 4.774. The remaining regret
gap is mostly from train/test distribution shift, not model quality.

Full breakdown in [docs/RESULTS.md](rtb-bid-model/docs/RESULTS.md).

---

## Dataset

iPinYou Season 2 -- real RTB impression logs from June 2013.

- Training: 10.4M impressions (June 6-12, first 85% temporal)
- Validation: 1.8M impressions (last 15% temporal)
- Test: 2.5M impressions (June 13-15, leaderboard set)
- 5 advertisers, clearing prices 0-300 CNY fen

Temporal split so the model can't cheat by looking ahead.
Test set is a completely different time window.

---

## Architecture

Two model types sharing the same MLP backbone (512-256-128-64, LayerNorm,
GELU, dropout):

**MDN (Mixture Density Network):** Outputs K Gaussian mixture components
(weights, means, sigmas) in log-price space. Softplus + floor on sigma
to keep it from collapsing.

**DiscreteBins (DLF-style):** Softmax over integer price bins. Can be
uniform, quantile-spaced, log-spaced, or sqrt-spaced. Quantile-spaced
with 200 bins was the best single model.

**Ensemble:** Weighted blend of MDN and bins CDFs. The MDN adds sharp
peaks (even though it's miscalibrated alone), the bins model adds
accuracy. Together they beat either one.

**Bid optimizer:** Grid search over 500 candidates in [0.5, 300],
picks the bid maximizing `(V - b) * CDF(b)`. Grid search worked
better than Newton on this problem.

---

## Project Structure

```
rtb-bid-model/
  config.yaml              -- hyperparameters and data paths
  requirements.txt         -- Python dependencies
  src/
    config.py              -- loads config.yaml
    features.py            -- raw bz2 logs -> processed parquets
    dataset.py             -- in-memory dataset, batch iterator
    model.py               -- MDN + DiscreteBins architectures
    loss.py                -- NLL losses (MDN mixture, bins CE, smoothed)
    train.py               -- training loop (AMP, EMA, cosine LR, early stopping)
    evaluate.py            -- NLL, ANLP, PIT, KS, coverage, regret
    bid_optimizer.py       -- grid + Newton bid optimization
    export_onnx.py         -- ONNX export for C++ server
    __init__.py            -- package marker
    experiments/           -- cloud experiment scripts (.py + .sh)
  docs/
    RESULTS.md             -- final results summary
    experiments.md         -- full experiment log (all training rounds)
    ml_journey.md          -- complete ML development history
    model.md               -- architecture details
    data.md                -- dataset and feature engineering documentation
    pipeline.md            -- training, evaluation, and ONNX export pipeline
    future.md              -- known limitations and planned improvements
    build_plan.md          -- Phase 2/3 build plan
    infrastructure.md      -- infrastructure stack design
  exports/                 -- checkpoints, ONNX, eval results (not in git)
  data/                    -- raw + processed data (not in git)
  results/plots/           -- diagnostic plots (not in git)
```

---

## Quick Start

```bash
conda create -n rtb python=3.11
conda activate rtb
pip install -r rtb-bid-model/requirements.txt
cd rtb-bid-model

# 1. Feature engineering (raw bz2 -> processed parquets)
PYTHONPATH=src python src/features.py

# 2. Train best single model (200 quantile-spaced bins)
PYTHONPATH=src python src/train.py --model bins --name bins_quant200 \
  --hidden 512,256,128,64 --batch_size 8192 --num_bins 200 \
  --lr 1e-3 --dropout 0.1 --weight_decay 5e-4 --ema_decay 0.99 \
  --seed 42 --epochs 5

# 3. Evaluate
PYTHONPATH=src python src/evaluate.py --ckpt exports/bins_quant200/best.pt \
  --name bins_quant200 --split test

# 4. Export to ONNX
PYTHONPATH=src python src/export_onnx.py --ckpt exports/bins_quant200/best.pt \
  --out exports/best_model.onnx --feature_config exports/feature_config.json
```

---

## ONNX Export

The exported model takes encoded features and outputs bin probabilities
(softmax is baked into the graph):

```
Inputs:
  cat    [B, 9]    int64    -- encoded categoricals
  cont   [B, 9]    float32  -- continuous + cyclical + binary
  tags   [B, 10]   int64    -- user tag indices (padded)

Output:
  probs  [B, num_bins]  float32  -- probability per price bin

Opset: 17, dynamic batch axis
```

Encoding details in `exports/feature_config.json`.

---

## Key Findings

1. **You need distributional prediction.** Point estimates can't optimize
   first-price bids. LightGBM regression was worse than bidding naively.

2. **Discrete bins beat Gaussian MDN.** Non-parametric bins learn the
   actual price shape without assumptions. 20.17 vs 21.21 fen.

3. **Quantile-spaced bins beat uniform.** More resolution where prices
   are dense. 20.17 vs 20.33 fen.

4. **A bad model can help an ensemble.** The overfit MDN (33.13 alone)
   has sharp peaks that help the bid optimizer when blended with the
   more accurate bins model.

5. **Calibration != bid quality.** Better-calibrated models had worse
   regret. The optimizer wants sharp peaks, not smooth distributions.

6. **Dropout matters more than architecture.** Changing 0.05 to 0.02
   recovered 12 fen. Wider/deeper networks didn't help.

---

## Known Limitations

- **Selection bias:** training data only has auctions iPinYou won
- **No CTR model:** impression value V is fixed at 150 fen (no click
  data in training set)
- **Temporal ceiling:** train/test distribution shift caps performance
- **First-price framing on second-price data:** the Go feedback loop
  will generate true first-price outcomes

See [docs/future.md](rtb-bid-model/docs/future.md) for what's planned.

---

## What's Next

Phase 2: C++ inference server with gRPC, ONNX Runtime, thread pool,
micro-batching, circuit breaker, and Prometheus metrics. Multiple
instances run the same model with different strategies and compete
in simulated auctions.

Phase 3: Go auction engine that runs the full exchange simulation
with a sliding window feedback loop for continuous model improvement.

See [docs/build_plan.md](rtb-bid-model/docs/build_plan.md) and
[docs/infrastructure.md](rtb-bid-model/docs/infrastructure.md).

---

## References

- Ren et al. (2019). *Deep Landscape Forecasting for Real-time Bidding Advertising* (KDD)
- Zhang et al. (2014). *Optimal Real-Time Bidding for Display Advertising*
- Zhu et al. (2017). *Winning Price Estimation in Real-Time Bidding*
- Cui et al. (2011). *Bid Landscape Forecasting in Online Ad Exchange Marketplace*

---

Mohammed Qureshi -- Carleton University
