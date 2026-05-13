#!/bin/bash
set -e
cd ~/rtb-bid-model
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

run() {
  name=$1
  shift
  echo "[runsweep] starting $name with: $@"
  rm -rf exports/$name
  python3 src/train.py --model mdn --name $name --hidden 512,256,128,64 \
    --batch_size 8192 --ent_bonus 0.005 --ema_decay 0.99 "$@" >/dev/null 2>&1
  # extract last lines from inner log
  tail -2 logs/train_$name.log
  echo "[runsweep] $name done"
}

# Sweep over lr, dropout, wd, sigma_floor, K, seed.
run mdn_b1  --lr 1e-3 --dropout 0.05 --weight_decay 5e-4 --sigma_floor 0.05 --seed 42 --epochs 5
run mdn_b2  --lr 1e-3 --dropout 0.02 --weight_decay 5e-4 --sigma_floor 0.05 --seed 42 --epochs 5
run mdn_b3  --lr 1e-3 --dropout 0.0  --weight_decay 5e-4 --sigma_floor 0.05 --seed 42 --epochs 5
run mdn_b4  --lr 5e-4 --dropout 0.05 --weight_decay 5e-4 --sigma_floor 0.05 --seed 42 --epochs 5
run mdn_b5  --lr 2e-3 --dropout 0.05 --weight_decay 5e-4 --sigma_floor 0.05 --seed 42 --epochs 5
run mdn_b6  --lr 1e-3 --dropout 0.05 --weight_decay 1e-4 --sigma_floor 0.05 --seed 42 --epochs 5
run mdn_b7  --lr 1e-3 --dropout 0.05 --weight_decay 1e-3 --sigma_floor 0.05 --seed 42 --epochs 5
run mdn_b8  --lr 1e-3 --dropout 0.05 --weight_decay 5e-4 --sigma_floor 0.02 --seed 42 --epochs 5
run mdn_b9  --lr 1e-3 --dropout 0.05 --weight_decay 5e-4 --sigma_floor 0.10 --seed 42 --epochs 5
run mdn_b10 --lr 1e-3 --dropout 0.05 --weight_decay 5e-4 --sigma_floor 0.05 --seed 42 --epochs 5 --K 20
run mdn_b11 --lr 1e-3 --dropout 0.05 --weight_decay 5e-4 --sigma_floor 0.05 --seed 42 --epochs 5 --K 6
run mdn_b12 --lr 1e-3 --dropout 0.05 --weight_decay 5e-4 --sigma_floor 0.05 --seed 7  --epochs 5
run mdn_b13 --lr 1e-3 --dropout 0.05 --weight_decay 5e-4 --sigma_floor 0.05 --seed 11 --epochs 5
run mdn_b14 --lr 1e-3 --dropout 0.05 --weight_decay 5e-4 --sigma_floor 0.05 --seed 99 --epochs 5
run mdn_b15 --lr 1e-3 --dropout 0.05 --weight_decay 5e-4 --sigma_floor 0.05 --seed 23 --epochs 5
echo "[runsweep] all done"
