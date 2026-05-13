#!/bin/bash
# Sequential after runbig: bins_v2 with non-uniform edges and EMD loss.
set -e
cd ~/rtb-bid-model
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

while pgrep -f "runbig.sh" > /dev/null; do sleep 30; done
echo "[runbins_v2] runbig done at $(date)"

python3 src/bins_v2.py --config_json exports/bins_v2_configs.json --log logs/bins_v2_sweep.log
echo "[runbins_v2] done at $(date)"
