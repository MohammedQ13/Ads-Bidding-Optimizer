#!/bin/bash
# Builds the C++ bidder + Go engine images (one-time) and runs the demo backend:
# four companies competing + the auction engine, with the engine's scoreboard API
# exposed on :9200. --restart unless-stopped so it auto-starts when the VM starts.
set -e
cd "$(dirname "$0")/.."

echo "=== building images (one-time, ~10-15 min) ==="
sed -i 's/-j2/-j/' cpp-bidder/Dockerfile || true
sudo docker build -t cpp-bidder ./cpp-bidder
( cd go-engine && sudo docker build -t rtb-engine . )

echo "=== (re)starting backend ==="
sudo docker network create rtbnet 2>/dev/null || true
for c in a b c d; do
  sudo docker rm -f company-$c 2>/dev/null || true
  sudo docker run -d --name company-$c --network rtbnet --network-alias company-$c \
    --restart unless-stopped cpp-bidder ./build/bid_server config/company_$c.yaml
done
sleep 6
sudo docker rm -f engine 2>/dev/null || true
sudo docker run -d --name engine --network rtbnet -p 9200:9200 \
  --restart unless-stopped rtb-engine

sleep 8
echo "=== backend up. /stats: ==="
curl -s http://localhost:9200/stats | head -c 200
echo ""
