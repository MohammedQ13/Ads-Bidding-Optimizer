package engine

import (
	"context"
	"errors"
	"math/rand"
	"testing"
	"time"

	"google.golang.org/grpc"

	pb "rtbengine/gen/bidding"
	"rtbengine/internal/competitors"
	"rtbengine/internal/config"
	"rtbengine/internal/stats"
)

// mockDSP is a fake company server. It returns a fixed bid, optionally after a
// delay (to test the deadline) or with an error (to test exclusion).
type mockDSP struct {
	bid   float32
	err   error
	delay time.Duration
}

func (m *mockDSP) GetBid(ctx context.Context, in *pb.BidRequest, opts ...grpc.CallOption) (*pb.BidResponse, error) {
	if m.delay > 0 {
		select {
		case <-time.After(m.delay):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if m.err != nil {
		return nil, m.err
	}
	return &pb.BidResponse{BidPrice: m.bid}, nil
}

func (m *mockDSP) NotifyOutcome(ctx context.Context, in *pb.AuctionOutcome, opts ...grpc.CallOption) (*pb.Ack, error) {
	return &pb.Ack{Ok: true}, nil
}

func (m *mockDSP) Check(ctx context.Context, in *pb.HealthCheckRequest, opts ...grpc.CallOption) (*pb.HealthCheckResponse, error) {
	return &pb.HealthCheckResponse{}, nil
}

// helper to build an engine with given companies (id -> client, budget) and no
// synthetic competitors unless competitorMedian > 0.
func mkEngine(comps map[string]*mockDSP, budgets map[string]float64, V float64, deadlineMs, compCount int, compMedian float64) *Engine {
	cfg := &config.Config{}
	cfg.Auction.ImpressionValue = V
	cfg.Auction.DeadlineMs = deadlineMs
	cfg.Auction.DailyBudget = 1e9
	cfg.Competitors.Count = compCount
	cfg.Competitors.BaseMedian = compMedian
	cfg.Competitors.Spread = 0.01 // tight so the test is deterministic

	st := stats.New()
	e := &Engine{
		cfg:  cfg,
		st:   st,
		pool: competitors.New(compCount, compMedian, 0.01, 1),
		rng:  rand.New(rand.NewSource(1)),
	}
	for id, m := range comps {
		b := cfg.Auction.DailyBudget
		if budgets != nil {
			if v, ok := budgets[id]; ok {
				b = v
			}
		}
		e.comps = append(e.comps, &company{id: id, client: m, budget: b, dailyBudget: b})
		st.SetBudget(id, b)
	}
	return e
}

func TestFirstPriceHighestWins(t *testing.T) {
	comps := map[string]*mockDSP{
		"low":  {bid: 50},
		"high": {bid: 100},
		"mid":  {bid: 80},
	}
	e := mkEngine(comps, nil, 150, 100, 0, 0)
	e.runAuction(context.Background())

	snap := e.st.Snapshot()
	if snap.Companies["high"].Won != 1 {
		t.Errorf("highest bidder should win: %+v", snap.Companies)
	}
	// first price: winner pays its bid, profit = V - bid = 150 - 100 = 50
	if snap.Companies["high"].Profit != 50 {
		t.Errorf("profit should be 50, got %f", snap.Companies["high"].Profit)
	}
	if snap.Companies["high"].Spend != 100 {
		t.Errorf("spend should be 100 (own bid), got %f", snap.Companies["high"].Spend)
	}
	if snap.Companies["low"].Lost != 1 || snap.Companies["mid"].Lost != 1 {
		t.Errorf("losers should be recorded: %+v", snap.Companies)
	}
	if snap.LastClearing != 100 {
		t.Errorf("clearing price should be the winning bid 100, got %f", snap.LastClearing)
	}
}

func TestDeadlineExclusion(t *testing.T) {
	comps := map[string]*mockDSP{
		"slow": {bid: 200, delay: 200 * time.Millisecond}, // would win, but too slow
		"fast": {bid: 60},
	}
	e := mkEngine(comps, nil, 150, 20, 0, 0) // 20ms deadline
	e.runAuction(context.Background())

	snap := e.st.Snapshot()
	if snap.Companies["slow"].NoBid != 1 {
		t.Errorf("slow company should be excluded (no_bid): %+v", snap.Companies["slow"])
	}
	if snap.Companies["fast"].Won != 1 {
		t.Errorf("fast company should win since slow was excluded: %+v", snap.Companies["fast"])
	}
}

func TestBudgetTooLowCannotWin(t *testing.T) {
	comps := map[string]*mockDSP{
		"rich": {bid: 90},
		"poor": {bid: 120}, // highest bid but cannot afford it
	}
	budgets := map[string]float64{"poor": 50} // less than its 120 bid
	e := mkEngine(comps, budgets, 150, 100, 0, 0)
	e.runAuction(context.Background())

	snap := e.st.Snapshot()
	if snap.Companies["poor"].Won != 0 {
		t.Errorf("poor company can't afford its bid, should not win: %+v", snap.Companies["poor"])
	}
	if snap.Companies["rich"].Won != 1 {
		t.Errorf("rich company should win instead: %+v", snap.Companies["rich"])
	}
}

func TestCompetitorCanWin(t *testing.T) {
	comps := map[string]*mockDSP{
		"a": {bid: 50},
		"b": {bid: 60},
	}
	// synthetic competitor bids around 100000 -> beats every company
	e := mkEngine(comps, nil, 150, 100, 1, 100000)
	e.runAuction(context.Background())

	snap := e.st.Snapshot()
	if snap.Companies["a"].Won != 0 || snap.Companies["b"].Won != 0 {
		t.Errorf("no company should win when a competitor bids highest: %+v", snap.Companies)
	}
	if snap.Companies["a"].Lost != 1 || snap.Companies["b"].Lost != 1 {
		t.Errorf("both companies should be recorded as losing: %+v", snap.Companies)
	}
}

func TestBudgetResetsPerDay(t *testing.T) {
	// day = 2 auctions, budget = 100, company bids 100 (so one win empties it).
	// It should win on each day-start auction (budget reset), and be unable to
	// afford a win the rest of the day. Over 4 auctions (2 days) -> 2 wins.
	comps := map[string]*mockDSP{"a": {bid: 100}}
	e := mkEngine(comps, nil, 150, 100, 0, 0)
	e.cfg.Auction.DayAuctions = 2
	e.cfg.Auction.DailyBudget = 100
	for _, c := range e.comps {
		c.budget = 100
		c.dailyBudget = 100 // the per-instance cap the daily reset restores
	}
	for i := 0; i < 4; i++ {
		e.runAuction(context.Background())
	}
	snap := e.st.Snapshot()
	if snap.Companies["a"].Won != 2 {
		t.Errorf("with daily budget resets the company should win twice (day starts), got %d", snap.Companies["a"].Won)
	}
}

func TestReplaySettleIndependent(t *testing.T) {
	// in replay/backtest each company is scored independently against the fixed
	// clearing price (not a single-winner auction)
	comps := map[string]*mockDSP{
		"win":  {bid: 100}, // >= clearing 80 -> wins
		"lose": {bid: 60},  // <  clearing 80 -> loses
		"err":  {err: errors.New("down")},
	}
	e := mkEngine(comps, nil, 150, 100, 0, 0)
	req := e.makeRequest()
	e.replayOne(context.Background(), req, 80.0, 150.0, 100*time.Millisecond)

	snap := e.st.Snapshot()
	if snap.Companies["win"].Won != 1 || snap.Companies["win"].Profit != 50 {
		t.Errorf("winner should win with profit 50: %+v", snap.Companies["win"])
	}
	if snap.Companies["lose"].Lost != 1 {
		t.Errorf("low bidder should lose: %+v", snap.Companies["lose"])
	}
	if snap.Companies["err"].NoBid != 1 {
		t.Errorf("erroring company should be no_bid: %+v", snap.Companies["err"])
	}
	if snap.Auctions != 1 || snap.LastClearing != 80 {
		t.Errorf("auction/clearing not recorded: %+v", snap)
	}
}

func TestErroringCompanyExcluded(t *testing.T) {
	comps := map[string]*mockDSP{
		"broken": {err: errors.New("boom")},
		"ok":     {bid: 70},
	}
	e := mkEngine(comps, nil, 150, 100, 0, 0)
	e.runAuction(context.Background())

	snap := e.st.Snapshot()
	if snap.Companies["broken"].NoBid != 1 {
		t.Errorf("erroring company should be no_bid: %+v", snap.Companies["broken"])
	}
	if snap.Companies["ok"].Won != 1 {
		t.Errorf("the working company should win: %+v", snap.Companies["ok"])
	}
}
