# Model Architecture

## First-Price Auction Framing

Most public RTB datasets (including iPinYou) come from the second-price
auction era, where bidding your true value is the dominant strategy.
The industry shifted to first-price auctions around 2019-2021. In a
first-price auction you pay exactly what you bid, so bidding your true
value means zero profit. You have to shade your bid below true value,
bidding just enough to win and not more.

We apply a first-price framing to the historical data. The observed
clearing prices (payprice) represent what competitors were willing to
pay. We predict the distribution of competitor bids and optimize our
bid to maximize first-price expected profit:

    E[profit] = (V - b) * P(max_competitor_bid < b)

This makes the optimization problem harder and more realistic than
second-price, where you just bid your value.

## Why Distributional Prediction

Most RTB models predict a single number (the expected clearing price).
But knowing the average is not enough. Markets are volatile: the
clearing price for a given user segment might average 80 fen but the
actual distribution ranges from 15 to 250 fen. Bidding 85 blindly
means overpaying on cheap auctions and losing profitable expensive ones.

You need the full distribution so you can compute:

    E[profit] = (V - bid) * P(win at bid)

where P(win at bid) = P(clearing_price < bid), which requires the CDF.

We proved this empirically: LightGBM regression (point estimate) scored
37.13 fen regret, worse than blindly bidding the mean every time
(28.74 fen). Distributional prediction is not optional.

## Source Files

The model architecture spans three source files:

- `src/model.py` - neural network classes (embeddings, backbone, output heads)
- `src/loss.py` - loss functions and probability computations (NLL, CDF, PDF)
- `src/bid_optimizer.py` - bid optimization algorithms (grid search, Newton)

## Classes in model.py

### CategoricalEmbeddings

Embedding layer for all categorical features plus user tags. Each
categorical feature gets its own `nn.Embedding` table. User tags are
handled separately because each impression has a variable number of
tags (up to 10).

**Constructor:** `CategoricalEmbeddings(vocab_sizes, emb_dims, tag_vocab_size, tag_emb_dim)`

- `vocab_sizes`: dict mapping feature name to vocab size (e.g. `{'region': 36, 'city': 201, ...}`)
- `emb_dims`: dict mapping feature name to embedding dimension (from config.yaml)
- `tag_vocab_size`: number of distinct tags in training data
- `tag_emb_dim`: dimension for tag embeddings (default 24)

Creates `nn.Embedding(vocab_size, emb_dim)` for each categorical
feature, stored in a `ModuleList`. Creates a separate
`nn.Embedding(tag_vocab_size, tag_emb_dim, padding_idx=0)` for tags.
Index 0 is reserved for padding/unknown in all embedding tables.

**Attributes:**

- `self.embs`: ModuleList of per-feature embedding tables
- `self.feat_names`: ordered list of feature names
- `self.tag_emb`: tag embedding table (padding_idx=0)
- `self.tag_dim`: tag embedding dimension
- `self.out_dim`: total output dimension = sum of all emb dims + tag_emb_dim

**forward(cat, tags):**

- `cat`: (B, num_cat_features) int64 - one column per feature
- `tags`: (B, 10) int64 - up to 10 tag IDs per sample, padded with 0

Steps:
1. Look up each categorical feature's embedding: `embs[i](cat[:, i])` for each feature i
2. Embed all tags: `tag_emb(tags)` gives (B, 10, tag_dim)
3. Build a mask `(tags != 0)` to ignore padding indices
4. Mean-pool tag embeddings: sum masked embeddings, divide by number of non-zero tags (clamped to min 1 to avoid division by zero)
5. Concatenate all feature embeddings + pooled tag vector into one (B, out_dim) tensor

Returns (B, out_dim) float tensor.

### MLPBackbone

The shared MLP trunk used by both model types. Each layer is
`Linear -> LayerNorm -> GELU -> Dropout`. LayerNorm (not BatchNorm)
was chosen because it works better with small effective batch sizes
during mixed-precision training.

**Constructor:** `MLPBackbone(in_dim, hidden, dropout)`

- `in_dim`: input dimension (embedding output + continuous features)
- `hidden`: list of hidden layer sizes, e.g. [512, 256, 128, 64]
- `dropout`: dropout rate (0.02 for MDN, 0.1 for bins)

Builds `nn.Sequential` of `len(hidden)` blocks.

**Attributes:**

- `self.net`: the sequential stack
- `self.out_dim`: output dimension = last element of hidden list

**forward(x):** pass input through the sequential stack, returns (B, out_dim).

### MDN (Mixture Density Network)

Outputs K Gaussian mixture components in log-price space. Each component
has a weight, mean, and standard deviation. The mixture defines a full
probability distribution over log(payprice).

**Constructor:** `MDN(vocab_sizes, emb_dims, tag_vocab_size, tag_emb_dim, num_continuous, hidden, dropout, K, sigma_floor)`

- `K`: number of Gaussian components (default 6, tested 4-12)
- `sigma_floor`: minimum sigma value (default 0.05) to prevent collapse

Contains:
- `self.emb`: CategoricalEmbeddings instance
- `self.cont_norm`: nn.Identity() (normalization is done in features.py)
- `self.backbone`: MLPBackbone(emb.out_dim + num_continuous, hidden, dropout)
- `self.head`: nn.Linear(backbone.out_dim, 3 * K) - outputs pi, mu, sigma for each component
- `self.K`, `self.sigma_floor`

**forward(cat, cont, tags):**

1. Embed categoricals + tags via self.emb
2. Concatenate with continuous features
3. Run through backbone
4. Split head output (3*K values) into three (B, K) tensors:
   - `pi_logits = out[:, :K]` - raw mixture weights (pre-softmax)
   - `mu = out[:, K:2*K]` - component means in log-price space
   - `log_sigma = out[:, 2*K:]` - component log-standard-deviations
5. Apply sigma activation: `sigma = softplus(log_sigma) + sigma_floor`
   - softplus ensures sigma > 0
   - sigma_floor prevents collapse to near-zero, which makes the NLL blow up

Returns tuple: `(pi_logits, mu, sigma)`, each (B, K).

The pi_logits are NOT softmaxed here, the loss function handles that
with log_softmax for numerical stability. When computing the CDF for
bid optimization, softmax is applied to get actual mixture weights.

### DiscreteBins

Softmax over discrete price bins. Each bin corresponds to an integer
price level in CNY fen. Inspired by Deep Landscape Forecasting
(Ren et al., KDD 2019). Simpler than MDN - no parametric assumptions,
just learn P(payprice == k) for each integer k.

**Constructor:** `DiscreteBins(vocab_sizes, emb_dims, tag_vocab_size, tag_emb_dim, num_continuous, hidden, dropout, num_bins)`

- `num_bins`: number of price bins (default 200 for quantile-spaced, or 301 for uniform 0-300)

Contains same embedding + backbone as MDN, but head is:
- `self.head`: nn.Linear(backbone.out_dim, num_bins)

**forward(cat, cont, tags):**

Same embedding + backbone flow as MDN. Returns raw logits (B, num_bins).
Softmax is applied externally, by the loss function during training,
and by BinsForExport during ONNX export.

The CDF for bid optimization is just `cumsum(softmax(logits))`.

### build_vocab_sizes(artifacts)

Utility function. Takes the artifacts dict from features.py and returns
a dict of `{feature_name: vocab_size}` for constructing embedding tables.
Each vocab size = `len(encoder) + 1` because index 0 is reserved for
unknown/rare values.

## Embedding Dimensions

| Feature        | Cardinality    | Embedding Dim |
|----------------|----------------|---------------|
| region         | ~35            | 16            |
| city           | ~200+ (clipped)| 24            |
| domain         | ~13K (clipped) | 32            |
| ad_exchange    | 3              | 4             |
| slot_width     | ~15            | 8             |
| slot_height    | ~10            | 6             |
| slot_visibility| 3              | 4             |
| slot_format    | 3              | 4             |
| advertiser_id  | 5              | 6             |
| user_tags      | ~14K           | 24            |

Tags are variable-length per sample (up to 10 per impression). Each
tag ID maps to an embedding vector, then all tag vectors are mean-pooled
into a single vector. Padding index 0 is masked out so it doesn't
affect the mean.

High-cardinality features (city, domain) are clipped: values appearing
fewer than min_count times in training are mapped to a shared rare token
(index 0). Thresholds: domain=100, city=500.

## Ensemble

Weighted average of multiple models' CDFs in probability space. The
best ensemble (19.908 fen) blends 3 MDN models and 2 bins models:

    CDF_ensemble = 0.49*mdn_s42 + 0.05*mdn_c1 + 0.04*mdn_c4
                 + 0.40*bins_quant200 + 0.02*bins_sqrt300

The MDN contributes sharp peaks in the CDF (even when miscalibrated
alone), while the bins model contributes accuracy. Together they beat
either type alone.

Key finding: a "bad" MDN with 33.13 fen regret alone still improved
the ensemble. Calibration is not the same as bid quality. The optimizer
needs sharp peaks, not smooth distributions.

The ensemble search used Dirichlet-random weight sampling over 200
trials on a 200K subset, followed by greedy refinement, then full
validation of top candidates.

## Loss Functions (loss.py)

All loss functions live in `src/loss.py`. One precomputed constant:
`LOG_2PI = math.log(2 * pi)` used in the Gaussian log-likelihood.

### mdn_nll(pi_logits, mu, sigma, target, ent_bonus=0.0, target_jitter=0.0)

Negative log-likelihood of the target under the Gaussian mixture,
averaged over the batch. This is the primary training loss for MDN.

Math:

    log p(y|x) = logsumexp_k [ log(pi_k) + log N(y; mu_k, sigma_k) ]
    NLL = -mean( log p(y|x) )

where:

    log N(y; mu_k, sigma_k) = -0.5 * ((y - mu_k)/sigma_k)^2 - 0.5*log(2*pi) - log(sigma_k)

Steps:
1. Compute `log_pi = log_softmax(pi_logits)` for numerical stability
2. If `target_jitter > 0`, add Gaussian noise to the target as regularization
3. Compute per-component log probability `log_comp`
4. Combine via `logsumexp(log_pi + log_comp)` across components
5. Return negative mean

Optional entropy bonus: if `ent_bonus > 0`, compute the entropy of
the mixture weights `H(pi) = -sum(pi * log(pi))` and subtract
`ent_bonus * H(pi)` from the loss. This encourages the model to use
all K components instead of collapsing to a single dominant component.

### mdn_log_prob(pi_logits, mu, sigma, target)

Same math as `mdn_nll` but returns per-sample log probabilities
instead of the batch mean. Shape: (B,). Used by evaluate.py for
ANLP computation.

### mdn_cdf(pi_logits, mu, sigma, x)

CDF of the Gaussian mixture at point(s) x.

    CDF(x) = sum_k pi_k * Phi( (x - mu_k) / sigma_k )

where Phi is the standard normal CDF, computed via:

    Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))

Supports two input shapes:
- `x`: (B,) - one point per sample, returns (B,)
- `x`: (B, T) - T points per sample (for grid search), returns (B, T)

The multi-point mode is used by the bid optimizer to evaluate the CDF
at all candidate bids in one vectorized call.

### mdn_pdf(pi_logits, mu, sigma, x)

PDF of the Gaussian mixture at point(s) x. Same signature as mdn_cdf.

    PDF(x) = sum_k pi_k * N(x; mu_k, sigma_k)

where N is the Gaussian probability density. Used by the Newton bid
optimizer to compute the derivative of expected profit (needed for the
Newton update step).

### discrete_bins_nll(logits, target_bin)

Standard cross-entropy loss. The target is the actual payprice in fen,
clamped to [0, num_bins-1]. Each integer price level is treated as a
class. This is the primary training loss for DiscreteBins.

    loss = cross_entropy(logits, target_bin)

### discrete_bins_smoothed_nll(logits, target_bin, sigma=1.0)

Cross-entropy with Gaussian-smoothed soft targets. Instead of a one-hot
target at the true bin, creates a Gaussian kernel centered on the true
bin index:

    weights[k] = exp(-0.5 * ((k - target_bin) / sigma)^2)

Normalized to sum to 1 across all bins. Then:

    loss = -mean( sum_k weights[k] * log_softmax(logits)[k] )

This way predicting bin 80 when the true price was 81 is penalized
less harshly than predicting bin 50. Tested with sigma=1.0 and 2.0,
but it didn't improve regret in practice (see experiments.md).

## Bid Optimization (bid_optimizer.py)

The bid optimizer takes a predicted CDF and finds the bid that maximizes
first-price expected profit:

    b* = argmax_b (V - b) * CDF(b)

where V is the impression value (150 fen default, or per-auction
bidding_price) and CDF(b) = P(clearing_price < b) = probability of
winning at bid b.

All functions in `src/bid_optimizer.py`. All work in batched mode,
optimizing bids for B impressions at once.

### grid_optimize_mdn(pi_logits, mu, sigma, V, n_candidates=500, b_max=300.0)

Grid search over candidate bids for MDN.

Steps:
1. Create `n_candidates` evenly spaced bids in [0.5, b_max]
   (skip 0 because log(0) is undefined for the MDN CDF)
2. Take log of each candidate (MDN works in log-price space)
3. Compute CDF at each candidate using the Gaussian mixture:
   `z = (log(b) - mu_k) / (sigma_k * sqrt(2))`, then `0.5*(1+erf(z))`
4. Weight by softmax(pi_logits) and sum across K components
5. Expected profit = `(V - b) * CDF(b)` for each candidate
6. Zero out profit where `b > V` (bidding more than value = guaranteed loss)
7. Pick argmax across candidates per sample

Returns: `(best_bid, best_profit, candidates, profit_grid)`

If no bid gives positive expected profit, returns bid=0 (skip auction).

### grid_optimize_bins(probs, V, n_candidates=300, b_max=300.0)

Grid search for DiscreteBins. Simpler than MDN because the CDF is
just the cumulative sum of the probability vector.

Steps:
1. CDF = cumsum(probs) along the bin dimension
2. Each bin index IS the bid price in fen (bin 50 = bid 50 fen)
3. Profit = `(V - bin_idx) * CDF[bin_idx]` for each bin
4. Zero out where `bin > V`
5. Pick argmax

Returns: `(best_bid, best_profit, bin_indices, profit_grid)`

The `n_candidates` and `b_max` args are unused, the function just
evaluates at every integer bin since there are only num_bins candidates.

### newton_optimize_mdn(pi_logits, mu, sigma, V, n_iters=25, n_starts=8, b_max=300.0)

Newton-Raphson optimizer for MDN. Multi-start with damped updates.

**Derivation:** The first-order optimality condition for
`E[profit] = (V-b)*F(b)` is:

    dE/db = -F(b) + (V-b)*f(b) = 0

Rearranging: `b = V - F(b)/f(b)`

This is a fixed-point equation. Newton's method iterates:

    b_{t+1} = V - CDF(b_t) / pdf(b_t)

The inner function `cdf_pdf(b)` computes both CDF and PDF at bid b:
- CDF: same Gaussian mixture formula as grid search
- PDF in log-space: `sum_k pi_k * N(log(b); mu_k, sigma_k)`
- Convert to linear-space PDF: `pdf_linear = pdf_log / b`
  (Jacobian of the log transform)

**Multi-start:** Tries `n_starts` initial bids spread from 5% to 95% of V.
Each start runs `n_iters` Newton iterations. The start with the highest
profit at convergence is selected.

**Damping:** Each update is `b = 0.5*b_old + 0.5*b_newton` to prevent
overshooting. NaN values (from zero PDF) are replaced with 0.5.

**Why grid is preferred:** Newton diverges on multimodal CDFs where the
profit function has multiple local maxima. The MDN mixture can produce
multimodal distributions, causing Newton to oscillate or converge to a
local minimum. Grid search always finds the global maximum because it
evaluates everywhere. Newton is kept as an alternative for comparison.

### newton_optimize_bins(probs, V, n_iters=15, n_starts=8, b_max=None)

Newton-Raphson for discrete bins. Same multi-start + damping approach.

The `cdf_pdf(b)` function handles fractional bids (Newton lands between
integer bins) by:
- CDF: linear interpolation between adjacent bins
- PDF: approximated as the slope between adjacent CDF values

Same Newton update: `b_new = V - CDF(b) / PDF(b)`.

### perfect_profit(payprice, V)

Oracle profit: what you'd earn with perfect knowledge of the
clearing price. If `V > payprice`, bid just above payprice and earn
`V - payprice`. Otherwise skip (earn 0).

    oracle = max(0, V - payprice)

Used as the baseline for computing regret.

### regret(actual_bid, actual_profit, payprice, V)

Per-impression regret = oracle profit minus realized profit.

You win if `bid >= payprice`, pay your bid, earn `V - bid`.
If you lose (`bid < payprice`), earn 0. Regret is the gap between
what you actually earned and what a perfect bidder would have earned.

    realized = win * (V - bid)  where win = (bid >= payprice)
    regret = oracle_profit - realized

Lower regret = better bidding. This is the primary metric for
comparing models.

## Why Not a Transformer

The v1 of this project used a BidTransformer (transformer encoder on
tabular features). It had convergence issues: sigma would collapse
or explode, and the self-attention was overkill for tabular data where
there is no sequential structure. The MLP backbone is simpler, trains
faster, and works better for this problem.

## Evaluation Metrics

- **Regret** (primary): how much profit we leave vs a perfect oracle.
  Lower is better. Regret = oracle_profit - model_profit per impression.
- **ANLP**: average negative log probability. Lower = better density fit.
  For MDN, includes a Jacobian correction from log-space to linear-space.
- **KS statistic**: max gap between predicted and empirical CDF.
  Lower = better calibration.
- **PIT histogram**: probability integral transform. CDF evaluated at
  true value. If calibrated, PIT ~ Uniform(0,1).
- **Coverage**: fraction of true prices within predicted intervals
  at each percentile threshold.

## Results Summary

| Model                         | Regret (V=150) | ANLP  | KS   |
|-------------------------------|----------------|-------|------|
| Naive (bid mean payprice)     | 28.74          | -    | -   |
| LightGBM regression (point)  | 37.13          | -    | -   |
| LightGBM 14-quantile         | 21.84          | 6.77  | 0.20 |
| Best MDN single (K=6, dp=0.02)| 21.21         | 4.52  | 0.21 |
| Best bins single (quant 200) | 20.17          | 3.74  | 0.18 |
| Best ensemble (3 MDN + 2 bins)| 19.91         | 3.79  | 0.24 |
