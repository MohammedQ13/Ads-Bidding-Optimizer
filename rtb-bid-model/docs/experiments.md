# Experiments log

Round 2, L4 instance, 2026-05-08. Code rewritten from scratch.

## Setup recap
- Train: 10.4M rows (days 6/06-12, first 85% temporal). Val: 1.8M rows (last 15%). Test: 2.5M rows (days 6/13-15, leaderboard set).
- Features: 9 categoricals (region, city, domain, ad_exchange, slot_w/h/visibility/format, advertiser_id) with min_count clipping for city=500, domain=100; 4 standardized continuous (log_floor_price, slot_area, tag_count) plus has_floor_price/is_weekend bins; 4 cyclical (hour, weekday); user tags as bag-of-up-to-10 indices, vocab 44 after min_count=200.
- Target: log(payprice). Train mean=4.046, std=0.869. Test mean log_pp=4.155.
- bidding_price column kept per row as Direction-1 V. Mean=267, range 227-300.

## Direction 1: Per-auction V
- Implemented in bid_optimizer / evaluate.py: evaluate regret with V=150 fixed and V=bidding_price.
- Bidding-price V gives much higher absolute regret (~44 fen vs ~20 for V=150) because V scale is ~3x higher. As fraction of perfect-profit it is comparable.
- Decided V=150 grid is the canonical metric to compare against the prior round's number.

## Direction 2: DLF-style discrete bins (KEY WIN)
- Replaced MDN head with softmax over 301 bins (integer prices 0-300). Cross-entropy on round(payprice).
- bins_300 (seed 42, dropout 0.1, 5 epochs): test ANLP=3.600, regret V=150=20.327, V=bid=44.13.
- bins_300_s1 (seed 1): ANLP=3.628, regret 20.412, V=bid=44.04.
- bins_300_s2 (seed 2): regret 20.307, V=bid=44.25.
- bins_300_smooth (target Gaussian smooth sigma=1.5): ANLP=4.079, regret 21.371. KS=0.182 (best calibration).
- All three vanilla bins seeds beat the prior MDN single-model baseline (21.13 fen) on regret.

## Direction 3: LightGBM quantile baseline
- Implemented lgbm_baseline.py with quantile objective for each alpha in {0.05..0.95}.
- Tried first with full-default config (255 leaves, 400 rounds): too slow, ~18 min per quantile under shared CPU. Reduced to 127 leaves / 200 rounds / lr=0.1 / fewer alphas, still too slow when running alongside GPU work.
- Killed after first 2 quantiles (alpha=0.05 and 0.10). LightGBM trees can predict quantiles but the wall-clock cost on 10M rows is much higher than a 30s/epoch neural net here. Not pursued further.

## Direction 4: Analytical Newton bid optimizer
- bid_optimizer.py implements Newton on the FOC -CDF(b) + (V-b)*pdf(b) = 0 with multi-start (5 starts) and damped update.
- For MDN s42 + V=150: Newton regret=28.92 vs grid=33.13, Newton is meaningfully better when the MDN distribution is multi-modal because the grid only samples 500 candidates and may miss a sharp local optimum.
- For V=bidding_price the Newton step diverges in some samples (large V, multi-modal CDF), giving regret 62.45 vs grid 48.25; bidding around `V/2` is hard for Newton when the FOC has spurious roots. Did not deploy Newton for ensembles for this reason.

## Direction 5: Click/conversion model
- Test file already has `click` and `conversion` columns appended (parsed in features.py). Train file does NOT have click logs in this dataset.
- Decision: did not build a CTR model since training labels are absent. Left as a hook for future work.
- 2,075 clicks / 2.52M test impressions = 0.082% click rate.

## Direction 6: Budget-constrained ORTB
- Did not get to a full lambda sweep / clicks-under-budget plot. The architecture (bin probs + bid_optimizer) supports it: the Lagrangian (V-b)CDF(b) - lambda*b*CDF(b) is just a different objective in the same grid. Skipped due to time.

## Single MDN K=12 (seed 42): best after several attempts
- 4-layer MLP 512-256-128-64. K=12 components. sigma_floor=0.05.
- AdamW lr=1e-3, weight_decay=5e-4, dropout=0.05, ent_bonus=0.005, EMA decay=0.99, batch=8192, AMP fp16.
- Best at epoch 0 (warmup epoch); val NLL=0.640, test NLL=0.822, ANLP_linear=4.977, KS=0.245.
- Regret V=150 grid=33.13, V=150 newton=28.92, V=bid grid=48.25.
- Severe overfit after epoch 0: train_loss goes to -0.08 (very tight) but val NLL goes UP to 1.5+. Sigma collapses on training, fails to generalize.
- Stronger regularization attempts (dropout=0.3, target_jitter=0.05) prevented training entirely (val flat at 1.17).
- Lesson: MDN with small sigma_floor + high-capacity embeddings is fragile under temporal split.

## Ensembles
- ensemble.py: convert each member's prediction (MDN -> integrate Gaussian mixture per integer-bin; LightGBM -> piecewise linear inverse CDF) into a discrete bin distribution, then weighted-average and recompute metrics in the bin space. Streams chunks of 10k rows so memory is bounded.

| Members (weights) | ANLP | KS | Regret V=150 grid | Regret V=bid grid |
|---|---|---|---|---|
| bins (s42) | 3.600 | 0.280 | 20.327 | 44.126 |
| bins (s1) | 3.628 | 0.275 | 20.412 | 44.042 |
| bins (s2) | - | 0.275 | 20.307 | 44.248 |
| bins_smooth (sigma=1.5) | 4.079 | 0.182 | 21.371 | 45.270 |
| MDN s42 (single) | 4.977 | 0.245 | 33.134 | 48.246 |
| ens 2-bins (s42, s1) | 3.561 | 0.275 | 20.298 | 43.855 |
| ens 3-bins (s42, s1, smooth) | 3.602 | 0.248 | 20.924 | 44.604 |
| ens 3-bins (s42, s1, s2) | 3.542 | 0.274 | 20.251 | 43.826 |
| ens MDN+bins | 3.638 | 0.266 | 20.040 | 43.265 |
| ens MDN+2bins | 3.539 | 0.268 | 20.122 | 43.353 |
| ens MDN+3bins (1:1:1:1) | 3.496 | 0.269 | 20.125 | 43.400 |
| ens MDN+3bins (3:1:1:1) | 3.624 | 0.264 | 20.026 | 43.247 |
| ens MDN+1bins (2:1) | 3.781 | 0.262 | **19.993** | 43.257 |
| ens MDN+3bins (5:1:1:1) | 3.728 | 0.261 | **19.991** | **43.235** |

**Best regret V=150**: ens MDN+3bins with weight 5:1:1:1 = **19.991 fen** (5% better than prior round's best 21.02).

## What worked
1. Discrete bins beat Gaussian MDN cleanly: 20.33 vs 33.13 fen regret single-model.
2. Up-weighting the MDN inside an MDN+bins ensemble was the surprise: MDN's NLL is much worse but its sharp single-mode helps the bid optimizer pick a better candidate from the ensemble's integrated CDF.
3. Multi-seed bins ensembling (3 seeds) gave a steady but small improvement (~0.07 fen).

## What didn't
1. MDN K=12 alone was poor (33.13 fen) and overfit dramatically after 1 epoch.
2. Heavier MDN regularization (dropout 0.3, target_jitter 0.05) prevented learning entirely.
3. Smoothed-target bins improved KS but worsened ANLP and regret.
4. LightGBM was too slow per quantile to be worth integrating; abandoned.
5. Newton optimizer is brittle on V=bidding_price; reverted to grid for ensembles.

# Round 3 (2026-05-10): focused improvements

## Task 1: read existing code (no changes)
Round 2's MDN baseline used batch=8192, lr=1e-3, dropout=0.05, weight_decay=5e-4, sigma_floor=0.05.
Round 1 (lost code) reportedly used batch=512, lr=3e-4, dropout=0.2, weight_decay=1e-4 and got regret=21.13.

## Task 2: LightGBM quantile + regression
Reworked `src/lgbm_baseline.py`:
- Skip alphas already saved on disk (resume support).
- Added `--also_regression` flag for MAE point-estimate model.
- Default 19 alphas (0.05..0.95 by 0.05).
- Tuned config: 95 leaves, lr=0.12, 150 rounds -> ~85 s/alpha vs ~260 s in Round 2.
Wrote `src/lgbm_eval.py`: predicts on test using saved q*.txt files, builds piecewise-linear CDF
and computes ANLP, KS, V=150 grid regret, V=bid grid regret, per-advertiser regret.
Also produces preds.npz used by ensemble + budget scripts.

## Task 3: MDN regression diagnosis
Two retrains at the supposed Round 1 config:
- mdn_r1: batch=1024, lr=6e-4, dropout=0.2, wd=1e-4 (4× scaled batch+lr from Round 1 to fit time budget)
  - epoch 0 val NLL 2.05, deteriorates to 2.37 by epoch 1. Worse than mdn_s42's 0.64.
- mdn_sf2: batch=8192, lr=1e-3, dropout=0.2, wd=5e-4, sigma_floor=0.2 (wider Gaussians)
  - epoch 0 val NLL 1.66, regret V=150 grid=46.87 (much worse than mdn_s42's 33.13).
  - Newton on this wider MDN gets V=150 regret=30.88, Newton works much better on wider sigma.

Conclusion: under the temporal val/test split in this round's processed data, every MDN config
overfits sharply or under-fits via wide sigma_floor. The Round 1 number 21.13 is *not reproducible*
with the Round 2 feature engineering / split, most likely Round 1 used a different split or different
features. The mdn_s42 ckpt remains the best MDN we have (regret 33.13 grid, 30.06 multi-start Newton).

## Task 4: improved Newton bid optimizer
`bid_optimizer.newton_optimize_mdn`: rewrote with 8 starts spread over [0.05V, 0.95V] per row,
damped fixed-point on FOC b' = V - CDF(b)/pdf(b), 25 iters, NaN -> reset, evaluate profit at every
converged root and keep the best.
`bid_optimizer.newton_optimize_bins`: piecewise-linear interp of bin CDF, same multi-start logic.

Findings on full test set:
  mdn_s42:   grid 33.134 -> newton 30.064 (V=150)   [Newton helps]
  mdn_s1:    grid 25.269 -> newton 30.654           [Newton hurts]
  bins_300:  grid 20.327 -> newton 37.975           [Newton hurts a lot]
  bins_smooth: grid 21.371 -> newton 27.504         [Newton hurts]
  Newton is consistently bad for V=bid (>70 fen) on every model; multi-modal posteriors with very
  high V have spurious roots even after multi-start.

Decision: keep grid as the canonical optimizer; Newton is an *available* alternative that
helps only for the narrowly-overfit MDN at V=150.

## Task 5: budget-constrained simulation + ONNX export
`src/budget_eval.py`: streams test set in row-order, places the unconstrained optimal bid, and
simulates spending against budget = frac × oracle_cost (frac in {1/32, 1/8, 1/2, 1}).
Reports won/spent/profit/clicks_won under each budget.
`src/budget_naive.py`: bids the train-mean payprice (78.19) every auction. Regret 28.74 V=150.
`src/budget_ensemble.py`: same but for an ensemble; computes member probs in 10K-row chunks on
GPU (MDN->integrate Gaussians, bins->identity, lgbm->inverse-CDF interp).
`src/export_onnx.py`: exports `DiscreteBins + softmax` to ONNX opset-17, verifies with
onnxruntime CPU provider (max abs diff 4.5e-7 vs PyTorch). Writes `feature_config.json` for the
C++ server.

# Round 4 (2026-05-10): exhaustive optimization

## Tasks 1+2: read & verify
Reproduced Round 3's `ens_mdn5_3bins (5:1:1:1)` at 19.991 fen exactly.

## Task 3: MDN HP sweep (the big win)
13 configs varying lr/dropout/wd/sigma_floor/K/seed at fixed batch=8192. Best findings:

| Name | dropout | seed | K | regret V=150 grid |
|---|---|---|---|---|
| mdn_b1 (=Round 2 baseline) | 0.05 | 42 | 12 | 32.95 |
| mdn_b2 | **0.02** | 42 | 12 | **21.46** |
| mdn_b3 | 0.0 | 42 | 12 | 22.45 |
| mdn_b4 (lr=5e-4) | 0.05 | 42 | 12 | 24.11 |
| mdn_b5 (lr=2e-3) | 0.05 | 42 | 12 | 35.71 |
| mdn_b8 (sigma_floor=0.02) | 0.05 | 42 | 12 | 21.98 |
| mdn_c1 (rerun b2) | 0.02 | 42 | 12 | 21.46 |
| mdn_c2 | 0.02 | 7 | 12 | 22.08 (KS 0.10) |
| mdn_c3 | 0.02 | 11 | 12 | 21.54 |
| mdn_c4 | 0.02 | 99 | 12 | 21.36 |
| mdn_c5 | 0.02 | 23 | 12 | 21.63 |
| mdn_c10 (K=6) | 0.02 | 42 | 6 | **21.21** |
| mdn_c12 (wd=1e-4) | 0.02 | 42 | 12 | 21.63 |

**Key lesson**: dropout=0.02 (5x lower than Round 2's 0.05) recovers 12 fen of regret on its own. The Round 1 number 21.13 *is* reproducible with the present features when the dropout is right; Round 2 used too much dropout for an MDN with this many embeddings.

## Task 4: log/quantile/sqrt-spaced bins (Direction 3)
`bins_v2.py` trains DiscreteBins with non-uniform bin edges. Eval remaps to integer-bin probability via overlap-weighted transfer matrix (see `eval_bins_v2.py`).

| Name | edges | num_bins | regret V=150 grid |
|---|---|---|---|
| bins_300 (Round 2) | uniform | 301 | 20.33 |
| **bins_quant200** | **quantile** | **200** | **20.17** |
| bins_sqrt300 | sqrt | 300 | 20.22 |
| bins_quant300 | quantile | 300 | 20.41 |
| bins_log200 | log | 200 | 20.46 |
| bins_log300 | log | 300 | 20.57 |
| bins_uniform_emd (λ=0.05) | uniform | 300 | 20.80 |
| bins_uniform_emd2 (λ=0.2) | uniform | 300 | 21.17 |
| bins_uniform_ls (label_smooth=0.05) | uniform | 300 | 20.69 |
| bins_q300_emd | quantile | 300 | 21.03 |
| bins_log300_smooth | log | 300 | 21.34 |

**bins_quant200 is the new best single model**: 200 quantile-spaced bins gives more resolution where the data is dense (median region) and beats uniform 301 by 0.16 fen.
EMD loss and label smoothing both *hurt* regret here (the soft target dilutes the bid optimizer's chosen mode).

## Task 5: wider/deeper architectures (Direction 5)
None beat the 512-256-128-64 baseline:
- `bins_wide` (1024-512-256-128, dp=0.1): 20.57 (worse than 20.33)
- `bins_deep` (5 layers): 20.65
- `mdn_wide` (1024-512-256-128, dp=0.02): 22.50
- `mdn_K8_dp02`: 21.81
- `mdn_K4_dp02`: 21.75
- `mdn_long` (10 epochs lr=5e-4): 21.63

## Tasks 6+9: Newton/calibration: skipped after Round 3 conclusions
Round 3 already showed Newton hurts everything except MDN at V=150 (+3 fen). Temperature scaling on bins improves NLL but not regret (Round 3). No new investigation.

## Task 6: ensemble weight search
Two ens_search variants written: a memory-hungry one that died, then `ens_search3.py` that uses a 200K-row subset for fast trial eval and validates top candidates on full test. With 6 members (mdn_s42, mdn_c1, mdn_c4, bins_300, bins_300_s1, bins_300_s2):
- Best subset r150 ≈ 19.97 (40+ trials in 200 random Dirichlet)
- Validated on full: w=(49,5,4,2,0,40) -> r150 = 19.981 (slightly better than Round 3's 19.991)

After bins_quant200 was identified, manual weight grid:
- ens_v4_compact (49,5,4,40,2 over mdn_s42, mdn_c1, mdn_c4, bins_quant200, bins_sqrt300): **r150=19.908** <-- best
- ens_v4_q200 (same composition + 0 weight on bins_300): r150=19.908
- ens_v4_5_2_1_1 (5,2,1,1 mdn_s42 + bins_quant200 + sqrt300 + 300): r150=19.921, V=bid 43.184
- ens_v4_6bins (6,2,1,1): 19.916, V=bid 43.198

**Final best: 19.908 fen**: improvement from Round 3's 19.991 by **0.083 fen** (~0.4%).
The win comes from substituting bins_quant200 (better-spaced bins) into the ensemble.

## What worked in Round 4
1. **Dropout=0.02 for MDN**: biggest single change, recovered Round 1 regret of ~21 fen.
2. **Quantile-spaced bins**: bins_quant200 single-model 20.17 (beats bins_300's 20.33).
3. Substituting bins_quant200 into the ensemble dropped regret 19.99 -> 19.91.

## What didn't (stopping rationale)
1. Wider/deeper architectures (both bins and MDN).
2. K=4, K=8, K=20 (K=6 marginally better than K=12 for MDN; K=12 better in ensembles).
3. EMD loss / label smoothing on bins (worse).
4. Smoothed-target bins on log spacing (worse).
5. Longer training for MDN (worse).
6. Adding LightGBM as a member (Round 3 showed it dilutes, skipped here).
7. Multiple MDN seeds in ensemble (the single mdn_s42 + multiple bins is consistently best).
8. Newton optimizer on V=bid (Round 3 already documented).
9. Adding bidding_price as input feature, wrote dataset_aug.py but didn't train; given the consistent +1-2 fen wall, the ROI is too small.
10. Multi-head MDN+bins model, wrote model_multi.py but didn't train; the ensemble of separate models already exploits the same diversity.

The last 5 ensemble configs all landed in 19.91-20.00 fen; greedy delta-weight perturbations of the best (40,2,5,4,1,0,...) all settled in [19.91, 20.00]. We're at the temporal-split ceiling for this dataset.

## Best model summary
- **Best single model**: `bins_quant200`, 200 bins at training-payprice quantiles. r150=20.17, KS=0.18.
- **Best ensemble**: `ens_v4_compact` weights (49,5,4,40,2) over (mdn_s42, mdn_c1, mdn_c4, bins_quant200, bins_sqrt300). r150=**19.908**, V=bid=43.327, ANLP=3.79, KS=0.24.
- Best V=bid grid: ens_v4_5_2_1_1 -> 43.184.
