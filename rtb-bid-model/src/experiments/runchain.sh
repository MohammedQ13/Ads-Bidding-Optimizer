#!/bin/bash
cd ~/rtb-bid-model
# Wait for runsweep2 to finish, then start bins sweep
while pgrep -f runsweep2 > /dev/null; do sleep 30; done
echo "[chain] runsweep2 done at $(date)"
nohup bash runbins.sh > logs/runbins.log 2>&1
echo "[chain] runbins done at $(date)"
