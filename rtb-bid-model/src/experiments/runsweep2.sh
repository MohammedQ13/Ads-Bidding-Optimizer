#!/bin/bash
# Focused sweep on the winning dropout=0.02. Vary seeds + a few small variants.
set -e
cd ~/rtb-bid-model
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

run() {
  name=$1
  shift
  echo "[runsweep2] starting $name with: $@"
  rm -rf exports/$name
  python3 src/train.py --model mdn --name $name --hidden 512,256,128,64 \
    --batch_size 8192 --ent_bonus 0.005 --ema_decay 0.99 "$@" >/dev/null 2>&1
  tail -2 logs/train_$name.log
  echo "[runsweep2] $name done"
}

# Vary seeds (dp=0.02 winning config)
run mdn_c1 --lr 1e-3 --dropout 0.02 --weight_decay 5e-4 --sigma_floor 0.05 --seed 42  --epochs 5
run mdn_c2 --lr 1e-3 --dropout 0.02 --weight_decay 5e-4 --sigma_floor 0.05 --seed 7   --epochs 5
run mdn_c3 --lr 1e-3 --dropout 0.02 --weight_decay 5e-4 --sigma_floor 0.05 --seed 11  --epochs 5
run mdn_c4 --lr 1e-3 --dropout 0.02 --weight_decay 5e-4 --sigma_floor 0.05 --seed 99  --epochs 5
run mdn_c5 --lr 1e-3 --dropout 0.02 --weight_decay 5e-4 --sigma_floor 0.05 --seed 23  --epochs 5
# dp=0.02 with sigma_floor variants
run mdn_c6 --lr 1e-3 --dropout 0.02 --weight_decay 5e-4 --sigma_floor 0.02 --seed 42 --epochs 5
run mdn_c7 --lr 1e-3 --dropout 0.02 --weight_decay 5e-4 --sigma_floor 0.10 --seed 42 --epochs 5
# dp=0.01 (between 0 and 0.02)
run mdn_c8 --lr 1e-3 --dropout 0.01 --weight_decay 5e-4 --sigma_floor 0.05 --seed 42 --epochs 5
# K variants with dp=0.02
run mdn_c9  --lr 1e-3 --dropout 0.02 --weight_decay 5e-4 --sigma_floor 0.05 --seed 42 --K 20 --epochs 5
run mdn_c10 --lr 1e-3 --dropout 0.02 --weight_decay 5e-4 --sigma_floor 0.05 --seed 42 --K 6  --epochs 5
# Longer training with dp=0.02
run mdn_c11 --lr 1e-3 --dropout 0.02 --weight_decay 5e-4 --sigma_floor 0.05 --seed 42 --epochs 10
# wd variants
run mdn_c12 --lr 1e-3 --dropout 0.02 --weight_decay 1e-4 --sigma_floor 0.05 --seed 42 --epochs 5
run mdn_c13 --lr 1e-3 --dropout 0.02 --weight_decay 1e-3 --sigma_floor 0.05 --seed 42 --epochs 5
echo "[runsweep2] all done"
