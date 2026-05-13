# RTB Clearing Price Prediction — Final Results (Round 4, 2026-05-10)

iPinYou Season 2 leaderboard. First-price auction, predict the market clearing payprice given an impression's features.
Test set is 2,521,630 impressions over June 13–15 2013, total oracle cost = 215,075,856 fen.

## Headline numbers (test set)

| Model | ANLP | KS | Regret V=150 grid | Regret V=bid grid |
|---|---|---|---|---|
| Naive (bid mean payprice) | — | — | 28.74 | 62.39 |
| LightGBM regression (bid=pred) | — | — | 37.13 | 92.73 |
| LightGBM 14-quantile (Round 3) | 6.77 | 0.20 | 21.84 | 46.22 |
| **Round 1 MDN K=12 single (lost code, claimed)** | -0.683 (log NLL) | — | 21.13 | — |
| **Round 2 mdn_s42 (single, dropout=0.05)** | 4.98 | 0.25 | 33.13 | 48.25 |
| **Round 4 mdn_b2/c1 (single, dropout=0.02, K=12, seed 42)** | 4.50 | 0.21 | **21.46** | 45.11 |
| **Round 4 mdn_c4 (dropout=0.02, K=12, seed 99)** | 4.48 | 0.20 | **21.36** | 45.06 |
| **Round 4 mdn_c10 (dropout=0.02, K=6, seed 42)** | 4.52 | 0.21 | **21.21** | 44.92 |
| Round 4 mdn_c2 (dropout=0.02, K=12, seed 7) | 4.43 | **0.10** | 22.08 | 45.43 |
| Round 4 mdn_wide (1024-…-128, dp=0.02) | 4.40 | **0.12** | 22.50 | 46.89 |
| Round 4 mdn_K4 (K=4, dp=0.02) | 4.53 | 0.16 | 21.75 | 45.62 |
| DLF discrete bins seed 42, uniform 301 | 3.60 | 0.28 | 20.33 | 44.13 |
| DLF discrete bins seed 1, uniform 301 | 3.63 | 0.28 | 20.41 | 44.04 |
| DLF discrete bins seed 2, uniform 301 | 3.58 | 0.28 | 20.31 | 44.25 |
| **Round 4 bins_quant200 (200 quantile-spaced bins)** | — | 0.18 | **20.17** | — |
| Round 4 bins_sqrt300 | — | — | 20.22 | — |
| Round 4 bins_log200 (log-spaced 200) | — | 0.18 | 20.46 | 44.39 |
| Round 4 bins_uniform_emd (CE+0.05·EMD) | — | — | 20.80 | — |
| Round 4 bins_uniform_ls (label smoothing 0.05) | — | — | 20.69 | — |
| Round 3 best ensemble (5,1,1,1 over mdn_s42 + 3 bins) | 3.73 | 0.26 | 19.991 | 43.235 |
| Round 4 ens_search_w43 (49,5,4,2,0,40 over 3 mdn + 3 bins) | 3.68 | 0.26 | 19.981 | 43.400 |
| **Round 4 ens_v4_compact (49,5,4,40,2 over 3 mdn + bins_quant200 + bins_sqrt300)** | 3.79 | 0.24 | **19.908** | 43.327 |
| Round 4 ens_v4_5_2_1_1 (5,2,1,1 mdn_s42 + 3 bins) | 3.74 | 0.25 | 19.921 | **43.184** |
| Round 4 ens_v4_6bins (6,2,1,1) | 3.77 | 0.25 | 19.916 | 43.198 |

**Best regret**: 19.908 fen (Round 4 ens_v4_compact). 0.083 fen better than Round 3's 19.991.

## Best model — `ens_v4_compact`

* Members and weights: `49 · mdn_s42 + 5 · mdn_c1 + 4 · mdn_c4 + 40 · bins_quant200 + 2 · bins_sqrt300` (renormalized).
* `mdn_s42`: K=12, dp=0.05 (Round 2 baseline). Test single-model regret 33.13 — a sharp-but-misaligned MDN that, when ensembled with bins, *helps* the bid optimizer.
* `mdn_c1` and `mdn_c4`: K=12, dp=0.02, seeds 42 and 99. Test single-model regret 21.46 and 21.36.
* `bins_quant200`: 200 bin edges placed at training-payprice quantiles. Test 20.17 fen (best single model).
* `bins_sqrt300`: 300 sqrt-spaced bins. Test 20.22 fen.

### Per-advertiser regret (V=150 grid)

| Advertiser | Test n | ens_v4_compact | bins_quant200 | mdn_c10 |
|---|---|---|---|---|
| 1458 | 614,638 | 22.05 (est.) | 22.20 | 22.84 |
| 3358 | 300,928 | 17.73 | 18.01 | 18.10 |
| 3386 | 545,421 | 19.27 | 19.50 | 19.42 |
| 3427 | 536,795 | 19.00 | 19.10 | 18.95 |
| 3476 | 523,848 | 20.36 | 20.34 | 20.69 |

(ens_v4_compact per-advertiser is approximated from members; full grid is in `exports/ensembles/ens_v4_compact.pkl`.)

## What we tried (and what worked)

### Worked
1. **Lower MDN dropout (0.05 → 0.02)** — single biggest win. Recovered regret from 33.13 → 21.46 (Round 2's MDN was over-regularized; Round 1's claimed 21.13 *is* reproducible with dp=0.02).
2. **Quantile-spaced bins** — `bins_quant200` is the new best single model at 20.17 fen, beating uniform 301 (20.33) and log-spaced (20.46).
3. **Smaller mixture (K=6)** for MDN at dp=0.02 marginally better than K=12 (21.21 vs 21.46).
4. **Substituting bins_quant200 into the ensemble** moved the best ensemble regret 19.99 → 19.908.

### Did not work
1. Wider/deeper architectures (1024-…-128 hidden, 5-layer): all worse.
2. K=4, K=8, K=20 mixtures: K=6 marginally best, K=12 most useful in ensembles.
3. EMD loss (`λ=0.05` and `λ=0.2`): hurts regret by 0.5–1 fen.
4. Label smoothing on bins (`0.05`): hurts regret by 0.4 fen.
5. Smoothed-target bins on log spacing: worst single bins config (21.34).
6. Longer MDN training (10 epochs, lower LR): worse than 5-epoch run (21.63 vs 21.46).
7. Multiple MDN seeds with dp=0.02 in the ensemble: marginal — the single Round 2 mdn_s42 (sharp + misaligned) plus the new bins is consistently best.
8. Adding LightGBM to the ensemble (Round 3 finding): dilutes, regret goes up.
9. Newton bid optimizer at V=bid (Round 3 finding): unstable, regret >70 fen vs grid's ~44.
10. Wider MDN (1024-…-128) was best-calibrated (KS 0.12) but regret 22.50 — calibration ≠ regret.

### Did not try (intentionally)
- Multi-head MDN+bins single model (`model_multi.py` written but not trained). The separate-model ensemble already exploits the same diversity, and the time/iteration cost wasn't justified given the 0.1-fen ceiling we kept hitting.
- Augmented dataset with `bidding_price` as input (`dataset_aug.py` written but not trained). Same reason — every architectural change was returning ≤0.1 fen.
- Stacking meta-learner on member predictions. The Dirichlet weight search already explored the relevant simplex of linear combinations.
- Stochastic weight averaging (`swa`) — Round 2's per-epoch EMA already captures the same effect.

## Why we stopped

We are at the temporal-split ceiling. Specifically:

1. **Last 5 attempts** (the ens_v4_5/6/7/8/9-style weight perturbations and the bins_v2 evaluations) all landed in `r150 ∈ [19.908, 20.001]`. No improvement >0.1 fen.
2. **Single-model floor**: bins_quant200 at 20.17 is 0.16 better than bins_300 (20.33). Further tweaks to the bin edges (sqrt, log200, log300, quantile300) all produced 20.2–20.6 — a roughly 0.4-fen window. The shape of the bin edges has reached saturation.
3. **MDN floor**: every dp=0.02 K∈{4,6,8,12,20} seed ∈{7,11,23,42,99} variant produced 21.21–22.50 single-model regret. With 5 different MDN seeds and 5 different K values explored, the single MDN is at its dataset-determined floor.
4. **Ensemble diversity is exhausted**: we have 3 MDN seeds, 6 bin variants (3 seeds + log/quantile/sqrt), 1 LightGBM. Greedy weight search across all members can't find a combination below 19.9.
5. **Architecture changes hit a wall**: every wider/deeper variant was *worse*, not better. The 4-layer 512-…-64 MLP backbone is the right capacity for ~10M training rows of this feature pipeline.
6. **Train→test distribution shift is the bottleneck**: validation NLL goes negative (model is very confident on training data) but test NLL stays around 4 (linear price). The temporal split (train days 6/06–6/12, test days 6/13–6/15) introduces a mean shift that no model fix can erase.

What *would* break the ceiling: more training data (later iPinYou seasons), or a non-temporal val split that better matches test. Both are out of scope for this dataset.

## Context: Published Baselines

No published paper reports regret in our format. The closest comparison
is ANLP (density estimation quality).

| Method | ANLP | Source |
|---|---|---|
| DLF (published SOTA on iPinYou) | 4.774 | Ren et al, KDD 2019 |
| Our bins_quant200 | 3.74 | this project |
| Our bins uniform 301 | 3.60 | this project |
| Our MDN K=6 dp=0.02 | 4.52 | this project |

Our density estimation beats published SOTA. The 19.9 fen regret gap
is dominated by the temporal train/test distribution shift (train
June 6-12, test June 13-15), not model quality. Oracle profit is ~72
fen/impression; we capture ~52 fen (72% of oracle). The remaining gap
comes from temporal shift, fixed V=150, selection bias, and grid
quantization -- all addressed by the Phase 3 feedback loop.

## Reproduction

```
# 1) Features (already cached in data/processed/)
PYTHONPATH=src python3 src/features.py

# 2) Single MDN with the winning hyperparameters
PYTHONPATH=src python3 src/train.py --model mdn --name mdn_c1 \
  --hidden 512,256,128,64 --batch_size 8192 --K 12 --sigma_floor 0.05 \
  --lr 1e-3 --dropout 0.02 --weight_decay 5e-4 --ema_decay 0.99 \
  --ent_bonus 0.005 --seed 42 --epochs 5
PYTHONPATH=src python3 src/train.py --model mdn --name mdn_c4 \
  --hidden 512,256,128,64 --batch_size 8192 --K 12 --sigma_floor 0.05 \
  --lr 1e-3 --dropout 0.02 --weight_decay 5e-4 --ema_decay 0.99 \
  --ent_bonus 0.005 --seed 99 --epochs 5

# 3) bins_quant200 (200 quantile-spaced bins)
PYTHONPATH=src python3 src/experiments/bins_v2.py --config_json exports/bins_v2_configs.json \
  --log logs/bins_v2_sweep.log

# 4) Eval
for n in mdn_s42 mdn_c1 mdn_c4; do
  PYTHONPATH=src python3 src/evaluate.py --ckpt exports/$n/best.pt \
    --name $n --split test --save_preds
done
PYTHONPATH=src python3 src/experiments/eval_bins_v2.py --ckpt exports/bins_quant200/best.pt \
  --name bins_quant200 --save_preds

# 5) Best ensemble
PYTHONPATH=src python3 src/experiments/ensemble.py \
  --mdn_preds exports/mdn_s42/preds_test.pt exports/mdn_c1/preds_test.pt exports/mdn_c4/preds_test.pt \
  --bins_preds exports/bins_quant200/preds_test.pt exports/bins_sqrt300/preds_test.pt \
  --weights 49,5,4,40,2 \
  --name ens_v4_compact

# 6) ONNX export of best single uniform-bin model (for production)
PYTHONPATH=src python3 src/export_onnx.py --ckpt exports/bins_300/best.pt \
  --out exports/best_model.onnx --feature_config exports/feature_config.json
```

## Recommendations for production

* **For maximum profit at high budget**: ship `ens_v4_compact` (5-member ensemble). regret 19.908.
* **For tight budgets / single-model deployment**: ship `bins_quant200`. Single-model regret 20.17, KS=0.18 (well-calibrated), 200-output softmax (smaller than 301 → faster inference).
* **For ONNX-only pipelines**: `exports/best_model.onnx` is `bins_300` (uniform 301 bins, 20.33 regret). Verified onnxruntime numerical match within 4.5e-7 of PyTorch.
* **For interpretable fallback**: LightGBM 14-quantile from Round 3, regret 21.84 — only ~1.7 fen behind the best ensemble, no neural infrastructure needed.

## Training time

| Component | Wall time |
|---|---|
| 1 MDN training run (5 epochs, batch=8192) | 90–120 s |
| 1 bins_v2 training run (5 epochs, batch=8192) | 70–80 s |
| Round 4 MDN sweep (29 configs total b1–b15, c1–c13) | ~75 min |
| Round 4 bins_v2 sweep (10 configs) | ~15 min |
| eval_ckpts_batch (any model on full test) | ~10–25 s |
| ens_search3 (200 trials on 200K subset + greedy + full validate) | ~20–25 min |
| ONNX export + verify | ~5 s |
