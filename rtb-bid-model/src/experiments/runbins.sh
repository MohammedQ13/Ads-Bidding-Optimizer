#!/bin/bash
# Bins sweep: try lower dropout, bigger models, label smoothing.
set -e
cd ~/rtb-bid-model
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

run() {
  name=$1
  shift
  echo "[runbins] starting $name with: $@"
  rm -rf exports/$name
  python3 src/train.py --model bins --name $name --hidden 512,256,128,64 \
    --batch_size 8192 --ema_decay 0.99 --num_bins 301 "$@" >/dev/null 2>&1
  tail -2 logs/train_$name.log
  echo "[runbins] $name done"
}

run bins_d1 --lr 1e-3 --dropout 0.02 --weight_decay 5e-4 --seed 42 --epochs 5
run bins_d2 --lr 1e-3 --dropout 0.0  --weight_decay 5e-4 --seed 42 --epochs 5
run bins_d3 --lr 1e-3 --dropout 0.05 --weight_decay 5e-4 --seed 42 --epochs 5
run bins_d4 --lr 1e-3 --dropout 0.02 --weight_decay 1e-4 --seed 42 --epochs 5
run bins_d5 --lr 1e-3 --dropout 0.02 --weight_decay 5e-4 --seed 7  --epochs 5
run bins_d6 --lr 1e-3 --dropout 0.02 --weight_decay 5e-4 --seed 11 --epochs 5
run bins_d7 --lr 1e-3 --dropout 0.02 --weight_decay 5e-4 --seed 99 --epochs 5
echo "[runbins] all done"
