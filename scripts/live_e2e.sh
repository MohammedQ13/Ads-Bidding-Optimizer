#!/usr/bin/env bash
# Automated live end-to-end test of the whole feedback loop.
#
# Boots the four C++ companies + the Go engine on a private docker network
# (no host ports, so it won't clash with anything), lets auctions run, then
# triggers a retrain and checks the companies hot-swap the new model. Tears
# everything down at the end. Exits non-zero if any assertion fails.
#
# Usage: bash scripts/live_e2e.sh
set -u

NET=rtb_e2e_net
VMODELS=rtb_e2e_models
VDATA=rtb_e2e_data
fail=0
note() { echo "[e2e] $*"; }
assert() { if [ "$1" = "1" ]; then note "PASS: $2"; else note "FAIL: $2"; fail=$((fail+1)); fi; }

cleanup() {
  note "cleaning up"
  docker rm -f e2e-a e2e-b e2e-c e2e-d e2e-engine >/dev/null 2>&1
  docker volume rm $VMODELS $VDATA >/dev/null 2>&1
  docker network rm $NET >/dev/null 2>&1
}
trap cleanup EXIT

note "building images if needed"
docker image inspect cpp-bidder:latest >/dev/null 2>&1 || docker build -t cpp-bidder:latest cpp-bidder
docker image inspect rtb-engine:latest >/dev/null 2>&1 || docker build -t rtb-engine:latest go-engine
docker image inspect rtb-retrainer:latest >/dev/null 2>&1 || docker build -t rtb-retrainer:latest retrainer

cleanup
docker network create $NET >/dev/null
docker volume create $VMODELS >/dev/null
docker volume create $VDATA >/dev/null

note "seeding the shared model volume"
docker run --rm -v $VMODELS:/models cpp-bidder:latest sh -c "cp models/bid_model.onnx /models/" >/dev/null

note "starting the four companies (live configs, watch on)"
for c in a b c d; do
  docker run -d --name e2e-$c --network $NET --network-alias company-$c \
    -v $VMODELS:/models cpp-bidder:latest \
    ./build/bid_server config/company_${c}_live.yaml >/dev/null
done
sleep 4

note "starting the engine"
docker run -d --name e2e-engine --network $NET -v $VDATA:/data rtb-engine:latest >/dev/null
note "letting auctions run for 15s"
sleep 15

# --- assertion 1: auctions are flowing ---
STATS=$(docker run --rm --network $NET alpine wget -qO- http://e2e-engine:9200/stats 2>/dev/null)
AUCTIONS=$(echo "$STATS" | grep -oE '"auctions":[0-9]+' | grep -oE '[0-9]+')
[ -n "$AUCTIONS" ] && [ "$AUCTIONS" -gt 0 ] && assert 1 "engine ran auctions ($AUCTIONS)" || assert 0 "engine ran auctions"

# --- assertion 2: at least one company won something ---
WON=$(echo "$STATS" | grep -oE '"won":[0-9]+' | grep -oE '[0-9]+' | sort -nr | head -1)
[ -n "$WON" ] && [ "$WON" -gt 0 ] && assert 1 "a company won auctions ($WON)" || assert 0 "a company won auctions"

# --- assertion 3: outcomes were logged for retraining ---
LINES=$(docker run --rm -v $VDATA:/data alpine sh -c "wc -l < /data/outcomes.jsonl" 2>/dev/null | tr -d ' ')
[ -n "$LINES" ] && [ "$LINES" -gt 100 ] && assert 1 "outcomes logged ($LINES lines)" || assert 0 "outcomes logged"

# --- assertion 4: model version starts at 1 ---
MV1=$(docker exec e2e-a sh -c "wget -qO- http://localhost:9100/metrics | grep '^bid_model_version'" 2>/dev/null | grep -oE '[0-9]+$')
assert "$([ "$MV1" = "1" ] && echo 1 || echo 0)" "model version starts at 1 (got $MV1)"

note "running one retrain round (writes a new model to the shared volume)"
docker run --rm --network $NET -v $VMODELS:/models -v $VDATA:/data rtb-retrainer:latest \
  python retrain.py --once --min_rows 500 --epochs 1 >/dev/null 2>&1
sleep 6

# --- assertion 5: the company hot-swapped the new model (version ticked) ---
MV2=$(docker exec e2e-a sh -c "wget -qO- http://localhost:9100/metrics | grep '^bid_model_version'" 2>/dev/null | grep -oE '[0-9]+$')
assert "$([ "$MV2" = "2" ] && echo 1 || echo 0)" "model hot-swapped, version 1->2 (got $MV2)"

# --- assertion 6: the server is still serving after the swap (zero downtime) ---
OK=$(docker exec e2e-a sh -c "wget -qO- http://localhost:9100/metrics | grep -c '^bid_request_total'" 2>/dev/null)
[ -n "$OK" ] && [ "$OK" -gt 0 ] && assert 1 "server still serving after swap" || assert 0 "server still serving after swap"

echo
if [ "$fail" = "0" ]; then
  note "ALL E2E ASSERTIONS PASSED"
  exit 0
fi
note "E2E FAILED: $fail assertion(s)"
exit 1
