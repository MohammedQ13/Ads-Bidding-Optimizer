#!/bin/bash
# Full experiment driver. Run from ~/rtb-bid-model.
set -e
cd ~/rtb-bid-model
mkdir -p logs exports results/plots
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

run_log() {
  name=$1
  cmd=$2
  echo "[runall] starting: $name"
  bash -c "$cmd" 2>&1 | tee logs/$name.log
  echo "[runall] finished: $name"
}

# 1) MDN baseline (3 seeds)
python3 src/train.py --model mdn --name mdn_s42 --seed 42 2>&1 | tee logs/train_mdn_s42.log
python3 src/train.py --model mdn --name mdn_s1  --seed 1  2>&1 | tee logs/train_mdn_s1.log
python3 src/train.py --model mdn --name mdn_s2  --seed 2  2>&1 | tee logs/train_mdn_s2.log

# 2) DLF-style discrete bins
python3 src/train.py --model bins --name bins_300 --num_bins 301 --seed 42 2>&1 | tee logs/train_bins_300.log

# 3) DLF-style with smoothed targets (more calibrated)
python3 src/train.py --model bins --name bins_300_smooth --num_bins 301 --smooth_sigma 1.5 --seed 42 2>&1 | tee logs/train_bins_300_smooth.log

# 4) LightGBM quantile baseline
python3 src/lgbm_baseline.py --out exports/lgbm_quant 2>&1 | tee logs/lgbm_quant.log

# Eval
for n in mdn_s42 mdn_s1 mdn_s2 bins_300 bins_300_smooth; do
  python3 src/evaluate.py --ckpt exports/$n/best.pt --name $n --split test --save_preds 2>&1 | tee logs/eval_$n.log
done
python3 src/evaluate_lgbm.py --in_dir exports/lgbm_quant --name lgbm_quant 2>&1 | tee logs/eval_lgbm_quant.log

echo "[runall] all done"
