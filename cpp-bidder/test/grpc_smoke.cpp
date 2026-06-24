// End to end gRPC smoke test. Connects to a running bid server, replays the
// golden fixture requests over the real GetBid RPC, and checks the returned
// bids match what the Python pipeline computed. Also exercises Check and
// NotifyOutcome. This validates the whole path: gRPC -> encode -> batch ->
// ONNX -> optimize -> response.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>

#include <grpcpp/grpcpp.h>
#include <nlohmann/json.hpp>
#include "bid_service.grpc.pb.h"

using json = nlohmann::json;
using grpc::Channel;
using grpc::ClientContext;
using namespace bidding;

static int failures = 0;
static void check(bool ok, const std::string& msg) {
  if (!ok) {
    failures++;
    std::printf("  FAIL: %s\n", msg.c_str());
  }
}

int main(int argc, char** argv) {
  std::string target = "localhost:50051";
  std::string fixture_path = "test/golden_fixture.json";
  if (argc > 1) target = argv[1];
  if (argc > 2) fixture_path = argv[2];

  std::ifstream f(fixture_path);
  if (!f.is_open()) {
    std::printf("cannot open fixture %s\n", fixture_path.c_str());
    return 2;
  }
  json fx;
  f >> fx;

  auto channel = grpc::CreateChannel(target, grpc::InsecureChannelCredentials());
  auto stub = BidService::NewStub(channel);

  // wait for the server to be reachable
  if (!channel->WaitForConnected(std::chrono::system_clock::now() +
                                 std::chrono::seconds(5))) {
    std::printf("could not connect to %s\n", target.c_str());
    return 2;
  }

  // health check
  {
    ClientContext ctx;
    HealthCheckRequest req;
    HealthCheckResponse resp;
    auto st = stub->Check(&ctx, req, &resp);
    check(st.ok(), "Check rpc failed");
    check(resp.status() == HealthCheckResponse::SERVING, "server not SERVING");
  }

  int n = 0;
  for (const auto& c : fx["cases"]) {
    const auto& r = c["request"];
    std::string id = r.value("request_id", "case" + std::to_string(n));
    double expect_bid = c["bids"]["150"]["bid"].get<double>();

    BidRequest req;
    req.set_request_id(id);
    req.set_region(r["region"].get<int>());
    req.set_city(r["city"].get<int>());
    req.set_domain(r["domain"].get<std::string>());
    req.set_ad_exchange(r["ad_exchange"].get<int>());
    req.set_slot_width(r["slot_width"].get<int>());
    req.set_slot_height(r["slot_height"].get<int>());
    req.set_slot_visibility(r["slot_visibility"].get<int>());
    req.set_slot_format(r["slot_format"].get<int>());
    req.set_advertiser_id(r["advertiser_id"].get<std::string>());
    req.add_user_tags(r["user_tags"].get<std::string>());  // server splits it
    req.set_slot_floor_price(r["slot_floor_price"].get<float>());
    req.set_impression_value(150.0f);
    req.set_timestamp(r["timestamp"].get<int64_t>());

    ClientContext ctx;
    BidResponse resp;
    auto st = stub->GetBid(&ctx, req, &resp);
    check(st.ok(), id + " GetBid rpc failed");
    check(resp.dsp_id() == "company-a", id + " wrong dsp_id");
    check(!resp.used_fallback(), id + " unexpectedly used fallback");
    check(std::fabs(resp.bid_price() - expect_bid) < 1e-3,
          id + " bid got " + std::to_string(resp.bid_price()) + " want " +
              std::to_string(expect_bid));
    n++;
  }

  // dedup: the same request_id within the TTL should come back from cache
  {
    BidRequest req;
    req.set_request_id("dedup-test-1");
    req.set_region(216);
    req.set_impression_value(150.0f);
    req.set_timestamp(20130606120000);
    BidResponse r1, r2;
    ClientContext c1, c2;
    stub->GetBid(&c1, req, &r1);
    stub->GetBid(&c2, req, &r2);
    check(r1.bid_price() == r2.bid_price(), "dedup bids differ");
  }

  // outcome notification
  {
    ClientContext ctx;
    AuctionOutcome out;
    out.set_request_id("r0");
    out.set_won(true);
    out.set_clearing_price(95.0f);
    out.set_your_bid(99.0f);
    out.set_profit(51.0f);
    Ack ack;
    auto st = stub->NotifyOutcome(&ctx, out, &ack);
    check(st.ok() && ack.ok(), "NotifyOutcome failed");
  }

  if (failures == 0) {
    std::printf("GRPC SMOKE OK: %d requests, bids match fixture\n", n);
    return 0;
  }
  std::printf("GRPC SMOKE FAILED: %d checks failed\n", failures);
  return 1;
}
