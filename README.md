# RTB Bid Optimization — Distributional Clearing Price Prediction

A neural network that predicts the full probability distribution over clearing prices in real-time bidding auctions, trained with Negative Log-Likelihood loss. Built for COMP 4107 (Neural Networks) at Carleton University.

---

## Results

### Distributional vs Point Estimation (Held-Out Test Set)

Measured on the iPinYou Season 2 test set (June 13–15 2013, ~2.5M impressions). MLP baseline uses identical features and embeddings, trained with MSE loss on `log(payprice)`.

| Metric | MLP Baseline (MSE) | BidTransformer (NLL) |
|--------|--------------------|----------------------|
| Best Val Loss | 0.4461 (epoch 3) | -0.2949 (epoch 1) |
| Test NLL | — | 1.7735 |
| Mean Bid Regret | 24.88 fen | 25.39 fen |
| Naive Baseline Bid Regret | 40.85 fen | 40.85 fen |
| Auctions where Transformer beats MLP | — | 30.4% |

Both models substantially outperform the naive average-price baseline (40.85 fen mean regret). The Transformer beats the MLP on 30.4% of auctions — concentrated in high-volatility segments where uncertainty-aware bidding provides the most benefit. The MLP has lower mean regret overall due to the Transformer's early stopping behavior discussed in limitations.

### Calibration — Predicted Percentile vs Empirical Frequency

A well-calibrated model's predicted 70th percentile should contain ~70% of observed clearing prices.

| Predicted Percentile | Empirical Coverage | Error |
|---------------------|--------------------|-------|
| 10th | 6.6% | -3.4% |
| 20th | 14.7% | -5.3% |
| 30th | 24.4% | -5.6% |
| 40th | 46.4% | +6.4% |
| 50th | 58.9% | +8.9% |
| 60th | 66.8% | +6.8% |
| 70th | 74.5% | +4.5% |
| 80th | 82.6% | +2.6% |
| 90th | 91.4% | +1.4% |

Upper percentiles (70th–90th) are well-calibrated (≤4.5% error). Lower percentiles systematically undercover, reflecting the discrete price spike structure at round numbers in the dataset — the model tightens its distribution around common price points but underestimates probability mass in the low tail.

### Per-Advertiser KS Test (Log-Normal Assumption Validation)

| Advertiser | Test Impressions | KS Statistic | p-value | Result |
|-----------|-----------------|--------------|---------|--------|
| 1458 | 614,638 | 0.1510 | 0.0000 | Reject |
| 3358 | 300,928 | 0.2010 | 0.0000 | Reject |
| 3386 | 545,419 | 0.1595 | 0.0000 | Reject |
| 3427 | 536,793 | 0.1677 | 0.0000 | Reject |
| 3476 | 523,831 | 0.1819 | 0.0000 | Reject |

KS test formally rejects log-normality across all five advertisers. This is expected — discrete price spikes at round numbers (multiples of 10 CNY fen) create modes a smooth continuous distribution cannot fully capture. Log-normal remains the standard approximation in the RTB literature and produces practically useful calibration, particularly in the upper percentiles where bid decisions are most consequential.

### ONNX Export Equivalence Check

| Output | Max Absolute Diff | Mean Absolute Diff | Result |
|--------|-------------------|-------------------|--------|
| μ (mu) | 2.38e-6 | 4.3e-7 | PASS |
| σ (sigma) | 9.5e-7 | 7.0e-8 | PASS |

Verified across 1000 test inputs. Both outputs match PyTorch within 1e-4 tolerance. Dynamic axes verified for batch sizes {1, 32, 64} and tag sequence lengths {1, 5, 15, 50}.

---

## Overview

The standard approach to DSP bid optimization predicts a single clearing price and bids just above it:

```
Auction features → predict mean clearing price → bid mean + fixed margin
```

This ignores market volatility. The same impression type in this dataset clears anywhere from 4 to 287 CNY fen depending on competition. Bidding the mean overpays on cheap auctions and misses profitable ones.

This project predicts the full distribution:

```
Auction features → predict μ, σ of clearing price log-normal distribution
                 → compute analytically optimal bid maximizing
                   E[profit] = (V − b) · Φ_lognormal(b; μ, σ)
```

With a calibrated distribution you can compute a bid that explicitly trades off winning probability against profit margin. With only a point estimate you cannot.

---

## Project Structure

```
rtb-bid-model/
├── data/
│   ├── raw/                        # iPinYou Season 2 impression logs (not committed)
│   └── processed/
│       └── artifacts.pkl           # Label encoders, StandardScaler, tag vocab
├── src/
│   ├── features.py                 # Feature engineering pipeline
│   ├── dataset.py                  # PyTorch DataLoader (train/val/test)
│   ├── model.py                    # BidTransformer + MLPBaseline
│   ├── loss.py                     # Log-normal NLL loss
│   ├── bid_optimizer.py            # Analytical optimal bid computation
│   ├── train.py                    # Training loop, early stopping, checkpointing
│   ├── evaluate.py                 # Calibration curves, bid regret, KS test
│   └── export.py                   # ONNX export + equivalence check
├── exports/
│   ├── BidTransformer_best.pt      # Best Transformer checkpoint (epoch 1)
│   ├── MLPBaseline_best.pt         # Best MLP checkpoint (epoch 3)
│   ├── bid_model.onnx              # Exported ONNX model (opset 18)
│   └── feature_config.json         # Input spec: encoders, scaler, tag vocab
├── results/
│   └── plots/
│       ├── BidTransformer_calibration.png
│       ├── BidTransformer_learning_curves.png
│       ├── MLPBaseline_learning_curves.png
│       ├── bid_regret_comparison.png
│       └── bid_landscape_samples.png
├── notebooks/
│   └── eda.ipynb                   # EDA, log-normal validation, feature signal plots
├── requirements.txt
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.11, conda

### 1. Set up the environment

```bash
conda create -n rtb python=3.11
conda activate rtb
pip install -r requirements.txt
```

All commands run from the `rtb-bid-model/` directory.

### 2. Download the dataset

Download the iPinYou Season 2 dataset from Kaggle and place the impression logs in `data/raw/`:

```
https://www.kaggle.com/datasets/lastsummer/ipinyou
```

Expected files after extraction:

```
data/raw/
├── train.log.txt       # June 6–12 2013, 12,237,087 impressions
└── test.log.txt        # June 13–15 2013, ~2.5M impressions
```

### 3. Run the pipeline

```bash
# Feature engineering — encodes categoricals, scales floor price, generates cyclical time features
python src/features.py

# Validate DataLoader output — checks tensor shapes, dtypes, target ranges across all splits
python src/dataset.py

# Train BidTransformer (NLL loss) and MLP baseline (MSE loss)
python src/train.py

# Evaluate — calibration curves, bid regret, KS test, NLL on test set
python src/evaluate.py

# Export to ONNX and verify PyTorch vs ONNX outputs match within tolerance
python src/export.py
```

---

## Dataset

**iPinYou Season 2 — Real RTB Impression Logs (June 2013)**

| Metric | Value |
|--------|-------|
| Training impressions (June 6–12) | 12,237,087 |
| Train split | ~10.4M rows (85%) |
| Validation split | ~1.8M rows (15%) |
| Test impressions (June 13–15) | ~2.5M |
| Advertisers | 5 (IDs: 1458, 3358, 3386, 3427, 3476) |
| Payprice range | 0–300 CNY fen |
| Zero-price rows dropped | 175 |
| Missing user_tags filled | 1,691,615 (13.8%) |
| Global log-price mean (μ) | 4.0523 |
| Global log-price std (σ) | 0.8646 |

Train/validation split is temporal — validation contains strictly later auctions than training, preventing data leakage. Test set is a separate time window (June 13–15). Shuffling was explicitly avoided.

**Advertiser clearing price summary (training set):**

| Advertiser | Median Payprice (fen) |
|-----------|----------------------|
| 1458 | 60 |
| 3358 | 77 |
| 3386 | 67 |
| 3427 | 76 |
| 3476 | 73 |

**Log-normal assumption:** KS test on the full training set yields stat=0.1259, p≈0.0000, technically rejecting normality due to discrete price spikes at round numbers. Log-normal is used as the standard approximation, consistent with the RTB literature.

### Feature Engineering

| Feature Group | Features | Encoding |
|--------------|----------|----------|
| Categorical | region, city, ad_exchange, slot_width, slot_height, slot_visibility, slot_format, advertiser_id | Label encoded → embedding (dim=16, learned end-to-end) |
| Continuous | slot_floor_price | StandardScaler (mean=26.70, std=35.78) |
| Cyclical time | hour_sin, hour_cos, weekday_sin, weekday_cos | Sine/cosine encoding |
| User tags | Variable-length tag ID sequences | Mean-pooled embeddings |
| Target (train) | log(payprice) | Raw payprice retained for bid regret computation |

Key signals from EDA: `slot_format=5` (video/rich media) has median payprice ~150 vs ~65 for banner formats — the strongest single feature signal. `slot_visibility=1` (above-fold) commands a premium. `ad_exchange=1` median ~80 vs exchange 2 median ~55. Ad exchange distribution: exchange 3 = 4.4M rows, exchange 2 = 4.0M, exchange 1 = 3.9M.

DataLoader validation: cats `[512, 8]` int64, conts `[512, 5]` float32, tags `[512, variable]` int64 padded. Log-target range `[1.386, 5.659]` train, raw-target range `[4.0, 287.0]` train. All 3 splits pass all 20 feature validation checks.

---

## ML Components

### BidTransformer

A Transformer encoder applied to structured auction features. Self-attention learns which feature interactions predict competition level — for example, `slot_format` interacting with `ad_exchange`, or time-of-day interacting with `slot_visibility`.

```
Layer              Detail
────────────────────────────────────────────────────────────────────────
Input              8 categorical features (label-encoded int64)
                   + 5 continuous (1 StandardScaled floor price
                   + 4 cyclical time encodings)
                   + variable-length user_tags (int64, padded)
Embedding          dim=16 per categorical, learned end-to-end
Tag Embedding      Mean pooling over variable-length user tag sequences
Projection         Linear → hidden dim H=128
Transformer        3 encoder layers; 4 attention heads (dim=32 per head);
                   FFN dim=256; GELU activations; dropout=0.1;
                   residual connections + LayerNorm throughout
Output Head        Linear H=128 → 2 scalars: μ, σ (Softplus on σ to enforce σ > 0)
Parameters         424,770 total
```

**Training:** Adam optimizer, lr=1e-4, batch size=512, ~14.9 batches/sec on GPU. Early stopping patience=5 on validation NLL. Best checkpoint at epoch 1 (val NLL=-0.2949). Early stopping triggered at epoch 6.

Sigma statistics at epoch 1: mean=0.6182, min=0.0576, max=0.9706.

### MLPBaseline

A 3-layer MLP with ReLU activations and batch normalization, trained with MSE loss on `log(payprice)`. Identical feature inputs and embedding layers to BidTransformer — the only variables are the architecture and loss function, making the comparison clean.

```
Layer              Detail
────────────────────────────────────────────────────────────────────────
Input              Same embeddings and continuous features as Transformer
Hidden             3 layers: 256 → 128 → 64; ReLU; BatchNorm
Output Head        Linear 64 → 1 scalar (log price point estimate)
Parameters         88,385 total
```

**Training:** Same optimizer and batch size. Val MSE trajectory: epoch 1=0.4513, epoch 2=0.4497, epoch 3=0.4461 (best). No improvement epochs 4–8. Early stopping at epoch 8.

### Loss Function — Negative Log-Likelihood

```
L = log(σ) + (log(p) − μ)² / 2σ²
```

where `p` is the observed clearing price. Minimizing this is equivalent to maximizing the log-likelihood of the observed price under the predicted log-normal distribution. The model is penalized for both inaccurate means (wrong μ) and miscalibrated uncertainty — σ too small means overconfident and incurs high loss when the actual price is far from μ; σ too large is uninformative and also penalized. MSE only optimizes μ and produces no uncertainty estimate.

### Analytical Bid Optimization

Once μ and σ are predicted, the optimal first-price bid maximizes expected profit:

```
E[profit] = (V − b) · Φ_lognormal(b; μ, σ)
```

where `V` is the advertiser's impression value (150 fen) and `Φ_lognormal` is the log-normal CDF — the probability of winning at bid `b`. 500 candidate bids from 1 to 150 fen are evaluated and the argmax is returned. This is post-processing on the model's outputs, not a separate model.

**What σ does to the bid:** When σ is small (confident), the optimal bid clusters close to μ. When σ is large (uncertain), probability mass is spread out and the bid shades more conservatively — the model encodes its own uncertainty into the bidding decision. The MLP cannot do this because it produces no σ.

**Bid regret definition:** For an auction with known clearing price `p*`, regret = max possible profit (perfect information bid) − profit from model's bid. Lower = better. Naive baseline uses a fixed bid of 57.53 fen (historical average clearing price).

---

## ONNX Export

```
Inputs:
  cats   [batch_size, 8]             int64   — label-encoded categoricals
  conts  [batch_size, 5]             float32 — scaled continuous + cyclical time
  tags   [batch_size, tag_seq_len]   int64   — user tag IDs (variable length)

Outputs:
  mu     [batch_size]                float32 — log-mean of clearing price distribution
  sigma  [batch_size]                float32 — log-std of clearing price distribution

Opset: 18 (legacy TorchScript exporter, dynamo=False)
```

Feature config saved to `exports/feature_config.json`: 8 categorical encoders, tag vocab size=43, scaler mean=26.70, std=35.78.

---

## Validation Strategy

| Metric | What It Measures |
|--------|-----------------|
| Held-out NLL | Overall distribution quality on unseen test auctions. Primary model selection metric. |
| Calibration curve | Does predicted 70th percentile actually contain ~70% of observed prices? Measures whether the model's uncertainty estimate is trustworthy. |
| Bid regret | Profit gap between model's optimal bid and the theoretically perfect bid with full information. Connects distributional output to the economically meaningful outcome. |
| KS test (per advertiser) | Formally validates log-normal assumption per segment; identifies where it breaks down. |
| Bid landscape visualization | Predicted log-normal distribution as smooth curve, empirical clearing price histogram as overlay, computed optimal bid as vertical line. Makes calibration visually interpretable. |

---

## Limitations

**Transformer early stopping at epoch 1.** The best checkpoint was saved at epoch 1 because validation NLL degraded every subsequent epoch. This reflects temporal distribution shift — the validation window (June 11–12) contains auction patterns from later in the week that differ statistically from the training window (June 6–10). In a production setting this would be addressed with continuous fine-tuning on recent data. It is a known challenge with time-based splits on behavioral auction data.

**Log-normal assumption formally rejected by KS test.** Discrete price spikes at round numbers create modes a smooth continuous distribution cannot fully capture. Upper-percentile calibration is good (≤4.5% error for 70th–90th percentiles) despite the global rejection. A mixture model or non-parametric approach would fit better but adds inference complexity.

**Impression value V is a fixed constant.** The bid optimization formula requires an advertiser value estimate. This is set to 150 fen rather than learned from click or conversion signals, which are not present in the iPinYou dataset. A complete system would predict per-impression value from downstream conversion models.

**Dataset predates first-price auctions.** iPinYou was collected in 2013 under second-price auction mechanics. The first-price framing applied here is a reasonable approximation but the underlying bidding incentives differ from a dataset collected in a true first-price environment.

---

## References

- Zhu et al. (2017). *Winning Price Estimation in Real-Time Bidding*
- Cui et al. (2011). *Bid Landscape Forecasting in Online Advertising Auction Markets*
- Zhang et al. (2014). *Optimal Real-Time Bidding for Display Advertising*
- Zhang et al. (2016). *Deep Learning over Multi-field Categorical Data for Ad Click Prediction*
- Cai et al. (2017). *Real-Time Bidding by Reinforcement Learning in Display Advertising*

---

## Author

Mohammed Qureshi — Carleton University, COMP 4107 Neural Networks
