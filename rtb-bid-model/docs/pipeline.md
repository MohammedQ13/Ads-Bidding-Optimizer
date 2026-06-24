# Training and Evaluation Pipeline

End-to-end pipeline from raw data to exported ONNX model. Each stage
is a standalone Python script run from `rtb-bid-model/`.

```
config.yaml --> config.py (loads settings)
    |
    v
features.py (raw bz2 logs --> processed parquets + artifacts.pkl)
    |
    v
dataset.py (parquets --> BidDataset with CPU tensors)
    |
    v
train.py (BidDataset --> trained model checkpoint best.pt)
    |          uses: model.py (MDN / DiscreteBins), loss.py (NLL functions)
    v
evaluate.py (checkpoint + BidDataset --> metrics + regret)
    |          uses: bid_optimizer.py (grid / Newton optimizers)
    v
export_onnx.py (checkpoint --> ONNX model + feature_config.json)
```

All scripts use `PYTHONPATH=src` so imports between src files work.

---

## Configuration (config.py + config.yaml)

### config.py

Single function: `load_config(path=None)`. If path is None, it finds
`config.yaml` one directory above `src/` using `os.path.abspath(__file__)`.
Opens the YAML file, calls `yaml.safe_load`, returns the dict.

Every other script calls `load_config()` at startup.

### config.yaml Reference

**data section:**

| Field | Default | Description |
|-------|---------|-------------|
| raw_dir | `data/raw` | where raw bz2 files live |
| processed_dir | `data/processed` | output directory for parquets |
| target_col | `payprice` | the prediction target column |
| train_frac | 0.85 | fraction of training data for train split (rest is val) |
| train_files | list of 7 paths | bz2 files for June 6-12 |
| test_files | list of 1 path | bz2 file for June 13-15 |

**features section:**

| Field | Default | Description |
|-------|---------|-------------|
| categorical | list of 9 features | which columns get embeddings |
| embedding_dims | dict of 10 entries | dimension per feature (includes user_tags) |
| clip_min_count | `{city: 500, domain: 100}` | rare value threshold per feature |
| continuous | 4 features | log_floor_price, slot_area, tag_count, has_floor_price |
| cyclical | 4 features | hour_sin, hour_cos, weekday_sin, weekday_cos |
| binary | 1 feature | is_weekend |

**mdn section:**

| Field | Default | Description |
|-------|---------|-------------|
| hidden_layers | [512, 256, 128, 64] | MLP layer sizes |
| n_components | 6 | number of Gaussian mixture components (K) |
| dropout | 0.02 | dropout rate (was 0.05, reduced to 0.02 - recovered 12 fen) |
| sigma_floor | 0.05 | minimum sigma to prevent collapse |

**bins section:**

| Field | Default | Description |
|-------|---------|-------------|
| num_bins | 200 | number of discrete price bins (deployed model overrides to 301 for uniform integer-fen bins) |
| hidden_layers | [512, 256, 128, 64] | MLP layer sizes |
| dropout | 0.1 | dropout rate |
| price_min | 0 | minimum price bin |
| price_max | 300 | maximum price bin |

**training section:**

| Field | Default | Description |
|-------|---------|-------------|
| batch_size | 8192 | training batch size |
| max_epochs | 12 | maximum training epochs |
| learning_rate | 1e-3 | base learning rate for cosine schedule |
| weight_decay | 5e-4 | AdamW weight decay |
| early_stopping_patience | 3 | epochs without improvement before stopping |
| grad_clip | 1.0 | max gradient norm for clipping |
| warmup_steps | 500 | linear warmup steps before cosine decay |
| ema_decay | 0.99 | exponential moving average decay |
| seed | 42 | random seed |
| num_workers | 4 | (unused - data loaded all at once, not with DataLoader) |

**evaluation section:**

| Field | Default | Description |
|-------|---------|-------------|
| percentiles | [5, 10, 15, ..., 95] | thresholds for coverage table |
| impression_value | 150.0 | default V for regret computation |
| bid_candidates | 500 | number of grid search candidates |

---

## Training (train.py)

### CLI Arguments

All args have defaults from config.yaml. CLI overrides take priority.

| Arg | Type | Default | Description |
|-----|------|---------|-------------|
| `--model` | str | 'mdn' | 'mdn' or 'bins' |
| `--name` | str | 'mdn_v1' | run name (output dir + log file) |
| `--seed` | int | from config | random seed |
| `--epochs` | int | from config | max epochs |
| `--lr` | float | from config | base learning rate |
| `--batch_size` | int | from config | batch size |
| `--K` | int | from config | MDN mixture components |
| `--num_bins` | int | from config | number of discrete bins |
| `--smooth_sigma` | float | 0.0 | Gaussian label smoothing sigma (bins only) |
| `--hidden` | str | from config | comma-separated layer sizes e.g. '512,256,128,64' |
| `--dropout` | float | from config | dropout rate |
| `--ema_decay` | float | from config | EMA decay factor |
| `--ent_bonus` | float | 0.0 | entropy bonus for MDN component diversity |
| `--target_jitter` | float | 0.0 | noise added to MDN targets |
| `--weight_decay` | float | from config | AdamW weight decay |
| `--sigma_floor` | float | from config | MDN minimum sigma |

### Utility Functions

**set_seed(s):** Sets random seed for numpy, torch CPU, and all CUDA
devices. Called once at startup.

**cosine_lr(step, warmup, total, base_lr, min_lr_ratio=0.05):**
Learning rate schedule. Two phases:
- Warmup (steps 0 to `warmup`): linear ramp from 0 to base_lr.
  Avoids large updates when weights are still random.
- Cosine decay (steps `warmup` to `total`): cosine annealing from
  base_lr down to `base_lr * min_lr_ratio` (default 5% of base).

Called every training step. The LR is manually set on each param group.

**to_dev(d, device):** Moves all tensors in a dict to target device
with `non_blocking=True`. Used for transferring batches to GPU.

**evaluate_loss(model, ds, device, model_type, batch_size, num_bins=None):**
Computes average loss over a full dataset in eval mode with no_grad.
For MDN: returns mean mdn_nll. For bins: returns mean cross-entropy
on clamped integer targets. Used for validation after each epoch.

### EMA Class

Exponential Moving Average of model weights. Keeps a shadow copy of
all trainable parameters. After each optimizer step, the shadow is
blended with the current weights:

    shadow = decay * shadow + (1 - decay) * current_weights

Over time, the shadow converges to a smoothed version of the weights
that often generalizes better than the raw weights, especially when
training is noisy.

**Methods:**

- `__init__(model, decay=0.999)`: snapshot all parameters as initial shadow
- `update(model)`: blend current weights into shadow
- `copy_to(model)`: overwrite model parameters with shadow
- `state_dict()`: return cloned shadow dict (for checkpointing)

### Training Loop

The main function does the following:

1. **Setup:** Parse args, load config, merge CLI overrides. Set seed,
   detect CUDA. Load artifacts, build vocab sizes. Construct MDN or
   DiscreteBins model based on --model flag.

2. **Optimizer:** AdamW with configurable weight decay. GradScaler
   for mixed precision training.

3. **Per-epoch training loop:**
   - Iterate batches with `shuffle=True, drop_last=True`
   - Each step:
     a. Update LR via cosine_lr schedule
     b. Forward pass under `torch.amp.autocast('cuda', dtype=float16)`
     c. Compute loss:
        - MDN: `mdn_nll(pi, mu, sigma, log_pp, ent_bonus, target_jitter)`
        - Bins: `discrete_bins_nll(logits, pp_int)` or
          `discrete_bins_smoothed_nll(logits, pp_int, sigma)` if smooth_sigma > 0
     d. Scale loss with GradScaler, backward pass
     e. Unscale gradients, clip to max_norm=grad_clip
     f. Optimizer step, scaler update
     g. EMA update
   - Log every 100 batches

4. **Dual validation:** After each epoch, evaluate validation loss
   with both raw weights and EMA weights:
   - Compute val_loss_raw with the actual model weights
   - Temporarily swap in EMA shadow weights, compute val_loss_ema
   - Restore raw weights
   - Pick whichever loss is lower for checkpoint selection

5. **Checkpointing:** If val loss improves by at least 1e-5, save
   `exports/<name>/best.pt`. The better weights (raw or EMA) are
   saved. Checkpoint contains:

   | Key | Type | Description |
   |-----|------|-------------|
   | `ema_state` | dict | all EMA shadow parameters |
   | `val_loss` | float | best validation loss |
   | `use_ema` | bool | whether EMA weights were used |
   | `config` | dict | model architecture config (see below) |
   | `full_state_dict` | dict | model weights |

   The `config` dict in the checkpoint stores everything needed to
   rebuild the model without needing artifacts.pkl:

   | Field | Description |
   |-------|-------------|
   | model_type | 'mdn' or 'bins' |
   | K | number of MDN components (None for bins) |
   | num_bins | number of bins (None for MDN) |
   | hidden | list of hidden layer sizes |
   | dropout | dropout rate |
   | vocab_sizes | dict of {feature: vocab_size} |
   | emb_dims | dict of {feature: emb_dim} |
   | tag_vocab_size | tag vocabulary size |
   | num_continuous | number of continuous input features |
   | sigma_floor | MDN sigma floor (None for bins) |
   | tag_emb_dim | tag embedding dimension |

6. **Early stopping:** Counter `bad` increments when val loss doesn't
   improve. Training stops when `bad >= early_stopping_patience`.

### Output Files

- `exports/<name>/best.pt` - model checkpoint
- `logs/train_<name>.log` - training log (printed to both stdout and file)

---

## Evaluation (evaluate.py)

### CLI Arguments

| Arg | Type | Default | Description |
|-----|------|---------|-------------|
| `--ckpt` | str | required | path to model checkpoint |
| `--name` | str | required | run name (for output directory) |
| `--split` | str | 'test' | 'test' or 'val' |
| `--save_preds` | flag | False | save full prediction tensors |

### Functions

**load_model(ckpt_path, device):**
Loads checkpoint, reads the config dict, rebuilds MDN or DiscreteBins
with the saved architecture parameters, loads the state dict with
`strict=False`, moves to device, sets eval mode. Returns `(model, config)`.

**collect_predictions(model, ds, device, model_type, batch_size, num_bins=None):**
Decorated with `@torch.no_grad()`. Runs model on every batch and collects
outputs into a single dict of concatenated tensors.

For MDN, stores: pi_logits, mu, sigma, log_prob, log_pp, pp, bid.
For bins, stores: probs (softmax of logits), log_prob (log of predicted
probability at the true bin), log_pp, pp, bid.

The log_prob for bins is computed as:
`log(probs.gather(1, true_bin).clamp(min=1e-30))` - grab the predicted
probability at the true price index.

**compute_pit_mdn(preds):**
Probability Integral Transform for MDN. Evaluates the Gaussian mixture
CDF at each sample's true log_payprice. If the model is well calibrated,
the PIT values should be uniformly distributed on [0,1].

Uses the same erf-based CDF formula as mdn_cdf in loss.py.
Returns a numpy array.

**compute_pit_bins(preds):**
PIT for discrete bins. Cumulative sum of probabilities evaluated at
the true payprice bin index. Returns numpy array.

**ks_statistic(pit):**
Kolmogorov-Smirnov statistic: the maximum absolute difference between
the empirical CDF of PIT values and the uniform CDF. Measures how far
the model's predicted quantiles deviate from their expected frequencies.

Steps:
1. Sort PIT values
2. Compute empirical CDF: `arange(1, n+1) / n`
3. `d_plus = max(empirical_cdf - sorted_pit)`
4. `d_minus = max(sorted_pit - arange(0, n)/n)`
5. KS stat = max(d_plus, d_minus)

Lower is better. Perfect calibration gives 0.

**coverage_table(pit, pcts):**
For each percentile threshold p (e.g. 10, 20, ..., 90), computes the
fraction of PIT values <= p/100. Perfect calibration gives `p/100`
for each threshold. Used to generate the coverage table in results.

**regret_metrics_mdn_grid(preds, V_value, ...):**
Batched grid search regret for MDN. Processes the full dataset in
chunks of `batch_size=8192`. For each chunk: calls `grid_optimize_mdn`,
gets optimal bids, then calls `regret()` to compute per-impression regret.

V_value can be a number (e.g. 150.0) or the string 'bidding' to use
each impression's iPinYou bidding_price as V.

**regret_metrics_bins_grid(preds, V_value, ...):**
Same thing for discrete bins. Calls `grid_optimize_bins`.

**regret_metrics_mdn_newton(preds, V_value, ...):**
Same thing using Newton optimizer instead of grid search.

**regret_bins_newton(preds, V_value, ...):**
Newton optimizer regret for bins.

### Main Evaluation Flow

1. Load config, artifacts, and dataset (test or val split)
2. Load model checkpoint via load_model
3. Collect all predictions
4. **Density metrics:**
   - NLL: mean negative log probability
   - ANLP: average negative log probability in linear price space.
     For MDN, ANLP = -(log_prob - log_payprice).mean(). The subtraction
     of log_payprice is a Jacobian correction: the MDN models log-price,
     so converting the density to linear-price space requires dividing
     by the payprice (equivalently, subtracting log_payprice from the
     log probability). For bins, ANLP = NLL (already in linear space).
   - PIT: probability integral transform
   - KS: Kolmogorov-Smirnov statistic on PIT
   - Coverage: actual vs expected quantile coverage
5. **Bidding metrics (regret):**
   - Grid search at V=150 and V=bidding_price
   - Newton optimizer at V=150 and V=bidding_price
6. **Per-advertiser breakdown:** regret at V=150 (grid) split by
   advertiser ID. Reports count and mean regret for each.

### Output Files

- `exports/<name>/eval_<split>.pkl` - results dict with:

  | Key | Type | Description |
  |-----|------|-------------|
  | nll | float | negative log-likelihood |
  | anlp | float | average negative log probability (linear space) |
  | ks | float | KS statistic |
  | coverage | dict | {percentile: actual_coverage} |
  | regret_v150_grid | float | mean regret at V=150 (grid search) |
  | regret_vbid_grid | float | mean regret at V=bidding_price (grid) |
  | regret_v150_newton | float | mean regret at V=150 (Newton) |
  | regret_vbid_newton | float | mean regret at V=bidding_price (Newton) |
  | per_adv_regret_v150 | dict | {advertiser_id: mean_regret} |

- `exports/<name>/preds_<split>.pt` (if --save_preds) - full prediction
  tensors including model outputs, ground truth, PIT values, and
  click/conv labels (for test split).

---

## ONNX Export (export_onnx.py)

Exports a trained DiscreteBins model to ONNX format for the C++
inference server. MDN export is not implemented because bins was the
better single model and the C++ server only needs one model type.

### CLI Arguments

| Arg | Type | Default | Description |
|-----|------|---------|-------------|
| `--ckpt` | str | `exports/bins_300/best.pt` | path to model checkpoint |
| `--out` | str | `exports/best_model.onnx` | output ONNX file path |
| `--feature_config` | str | `exports/feature_config.json` | output config path |
| `--n_test` | int | 8 | batch size for dummy inputs / verification |

### BinsForExport Wrapper

A thin nn.Module that wraps DiscreteBins to bake softmax into the
ONNX graph. The raw DiscreteBins returns logits; BinsForExport applies
`F.softmax(logits, dim=-1)` so the ONNX model outputs probabilities
directly. The C++ server can use these probabilities for bid
optimization without any additional post-processing.

### Export Process

1. Load checkpoint, rebuild DiscreteBins with `dropout=0.0` (inference mode)
2. Wrap in BinsForExport
3. Create dummy inputs:
   - cat: (B, num_cat_features) int64 zeros
   - cont: (B, 9) float32 zeros
   - tags: (B, 10) int64 zeros
4. Reference forward pass with PyTorch for verification
5. `torch.onnx.export` with:
   - Input names: 'cat', 'cont', 'tags'
   - Output name: 'probs'
   - Dynamic axes on batch dimension (dim 0) for all inputs/outputs
   - Opset version 17
6. Verification: if onnxruntime is installed, load the ONNX model, run
   the same dummy inputs, check max absolute diff vs PyTorch reference.
   Previous runs showed max diff ~4.5e-7.

### ONNX Model Spec

```
Inputs:
  cat   [B, 9]    int64    - encoded categorical feature indices
  cont  [B, 9]    float32  - continuous + cyclical + binary features
  tags  [B, 10]   int64    - user tag indices (padded with 0)

Output:
  probs [B, N]    float32  - probability per price bin (sums to 1)

N = num_bins. The deployed model uses 301 uniform bins (bin index = price
in fen); the quantile-spaced research model used 200.
Opset: 17
Dynamic batch axis on dim 0
```

### feature_config.json Schema

Written alongside the ONNX model. Contains everything the C++ server
needs to encode raw ad request fields into the tensor format the
model expects.

| Field | Type | Description |
|-------|------|-------------|
| cat_order | list of str | ordered categorical feature names |
| cat_vocabs | dict of dicts | `{feature: {raw_value: index}}` for each feature |
| tag_vocab | dict | `{tag_string: index}` |
| cont_cols_order | list of str | ordered continuous column names (9 total) |
| cont_means | dict | `{col: mean}` for z-score standardization |
| cont_stds | dict | `{col: std}` for z-score standardization |
| tag_max_len | int | 10 (max tag sequence length) |
| tag_pad_idx | int | 0 (padding index for tags) |
| num_bins | int | number of output bins |
| embedding_dims | dict | `{feature: dimension}` for each embedding |
| tag_emb_dim | int | tag embedding dimension |

The C++ server reads this file at startup to build its in-memory
feature store (categorical lookup tables, normalization constants,
tag vocabulary).

---

## Reproduction Commands

```bash
cd rtb-bid-model

# 1. Feature engineering (takes ~10 min on 12M rows)
PYTHONPATH=src python src/features.py

# 2. Train the deployed single model (uniform bins, 301 integer-fen bins 0..300)
PYTHONPATH=src python src/train.py --model bins --name bins_300 \
  --hidden 512,256,128,64 --batch_size 8192 --num_bins 301 \
  --lr 1e-3 --dropout 0.1 --weight_decay 5e-4 --ema_decay 0.99 \
  --seed 42 --epochs 5

# 3. (Offline research only) train an MDN member for the ensemble study
PYTHONPATH=src python src/train.py --model mdn --name mdn_s42 \
  --hidden 512,256,128,64 --batch_size 8192 --K 6 \
  --lr 1e-3 --dropout 0.02 --weight_decay 5e-4 --ema_decay 0.99 \
  --sigma_floor 0.05 --seed 42 --epochs 5

# 4. Evaluate the deployed model
PYTHONPATH=src python src/evaluate.py \
  --ckpt exports/bins_300/best.pt \
  --name bins_300 --split test

# 5. Export to ONNX (defaults already point at bins_300)
PYTHONPATH=src python src/export_onnx.py \
  --ckpt exports/bins_300/best.pt \
  --out exports/best_model.onnx \
  --feature_config exports/feature_config.json
```

The deployed model is uniform `bins_300` (301 bins, bin index k = price k fen,
test regret 20.33), not the quantile-spaced `bins_quant200` that scored 0.16 fen
better offline (20.17). Quantile bins place the edges at training-payprice
quantiles, so the bin index no longer equals the price. Serving them correctly
would need the bin edges exported into feature_config.json plus an edge-aware
bid optimizer in the C++ server, neither of which was built. Uniform bins keep
the bin-index = price-in-fen invariant that the fixture, backtest, retrainer,
and C++ optimizer all assume, so they are the safe choice for deployment. See
experiments.md and RESULTS.md for the offline comparison.

---

## File Reference

| File | Role | Entry point? |
|------|------|--------------|
| `config.yaml` | all hyperparameters and data paths | no |
| `src/config.py` | YAML loader | no (imported by others) |
| `src/features.py` | raw data -> processed parquets + artifacts | yes: `python src/features.py` |
| `src/dataset.py` | processed parquets -> BidDataset tensors | no (imported by train/evaluate) |
| `src/model.py` | MDN + DiscreteBins neural networks | no (imported by train/evaluate/export) |
| `src/loss.py` | NLL losses, CDF, PDF functions | no (imported by train/evaluate) |
| `src/train.py` | training loop with AMP/EMA/cosine LR | yes: `python src/train.py` |
| `src/evaluate.py` | metrics: NLL, ANLP, KS, PIT, regret | yes: `python src/evaluate.py` |
| `src/bid_optimizer.py` | grid + Newton bid optimization | no (imported by evaluate) |
| `src/export_onnx.py` | ONNX export for C++ server | yes: `python src/export_onnx.py` |
| `src/__init__.py` | makes src/ importable as package | no (empty) |
| `src/experiments/` | cloud experiment scripts (.py + .sh) | yes (standalone) |
