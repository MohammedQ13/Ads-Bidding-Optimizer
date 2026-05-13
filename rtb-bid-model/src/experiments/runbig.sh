#!/bin/bash
# Bigger bins models + a bins model trained on log-spaced bins (via bins_v2).
set -e
cd ~/rtb-bid-model
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

run_train() {
  name=$1
  shift
  echo "[runbig] starting $name"
  rm -rf exports/$name
  python3 src/train.py "$@" --name $name >/dev/null 2>&1
  tail -2 logs/train_$name.log
  echo "[runbig] $name done"
}

# bigger bins: wider hidden 1024
run_train bins_wide --model bins --num_bins 301 --hidden 1024,512,256,128 --batch_size 8192 \
  --lr 1e-3 --dropout 0.1 --weight_decay 5e-4 --ema_decay 0.99 --seed 42 --epochs 5

# bigger bins: deeper 5 layers
run_train bins_deep --model bins --num_bins 301 --hidden 512,512,256,256,128 --batch_size 8192 \
  --lr 1e-3 --dropout 0.1 --weight_decay 5e-4 --ema_decay 0.99 --seed 42 --epochs 5

# bigger MDN with dp=0.02
run_train mdn_wide --model mdn --hidden 1024,512,256,128 --batch_size 8192 --K 12 --sigma_floor 0.05 \
  --lr 1e-3 --dropout 0.02 --weight_decay 5e-4 --ema_decay 0.99 --seed 42 --epochs 5

# MDN K=8 dp=0.02
run_train mdn_K8_dp02 --model mdn --hidden 512,256,128,64 --batch_size 8192 --K 8 --sigma_floor 0.05 \
  --lr 1e-3 --dropout 0.02 --weight_decay 5e-4 --ema_decay 0.99 --seed 42 --epochs 5

# MDN K=4 dp=0.02 (very simple mixture)
run_train mdn_K4_dp02 --model mdn --hidden 512,256,128,64 --batch_size 8192 --K 4 --sigma_floor 0.05 \
  --lr 1e-3 --dropout 0.02 --weight_decay 5e-4 --ema_decay 0.99 --seed 42 --epochs 5

# MDN dp=0.02 longer training (10 epochs)
run_train mdn_long --model mdn --hidden 512,256,128,64 --batch_size 8192 --K 12 --sigma_floor 0.05 \
  --lr 5e-4 --dropout 0.02 --weight_decay 5e-4 --ema_decay 0.99 --seed 42 --epochs 10

echo "[runbig] all done"
