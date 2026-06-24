// Package engine is the auction exchange. For every auction it makes up an
// impression, asks every C++ company for a bid (with a deadline), throws in some
// synthetic competitor bids, runs a first-price auction (highest bid wins and
// pays its own bid), tells each company whether it won, and logs the outcome so
// the retraining loop can learn from it.
package engine

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "rtbengine/gen/bidding"
	"rtbengine/internal/competitors"
	"rtbengine/internal/config"
	"rtbengine/internal/stats"
)

// company is one C++ bid server plus the budget we track for it.
type company struct {
	id          string
	client      pb.BidServiceClient
	metricsURL  string  // its Prometheus /metrics page, for scraping internals
	budget      float64 // remaining budget for the current simulated day
	dailyBudget float64 // the per-instance cap this resets to each day
	paced       bool    // true if sitting out the current auction for budget pacing
	mu          sync.Mutex
}

type Engine struct {
	cfg     *config.Config
	comps   []*company
	pool    *competitors.Pool
	st      *stats.Store
	rng     *rand.Rand
	logF    *os.File
	logW    *bufio.Writer
	logMu   sync.Mutex
	counter int64
}

// pools of plausible feature values so the made-up impressions look realistic.
// Some are real iPinYou values the model knows, some are unknown (cold start).
var regions = []int32{216, 80, 146, 3, 1, 134, 79, 275, 40, 124}
var cities = []int32{1, 79, 217, 2, 85, 275, 0, 334, 81, 95}
var domains = []string{"trqRTuiEMaYZ", "abcd1234", "mm_10027271", "unknownsite", "df_news"}
var advertisers = []string{"1458", "3358", "3386", "3427", "3476"}
var slotSizes = [][2]int32{{300, 250}, {728, 90}, {160, 600}, {300, 600}, {468, 60}}
var tagPool = []string{"10063", "10006", "10110", "10083", "10024", "10111", "10059", "10031"}

// logRecord is one auction written to the outcome log. It is the bid request
// plus what the auction cleared at - the label the retrainer learns from.
type logRecord struct {
	RequestID       string   `json:"request_id"`
	Region          int32    `json:"region"`
	City            int32    `json:"city"`
	Domain          string   `json:"domain"`
	AdExchange      int32    `json:"ad_exchange"`
	SlotWidth       int32    `json:"slot_width"`
	SlotHeight      int32    `json:"slot_height"`
	SlotVisibility  int32    `json:"slot_visibility"`
	SlotFormat      int32    `json:"slot_format"`
	AdvertiserID    string   `json:"advertiser_id"`
	UserTags        []string `json:"user_tags"`
	SlotFloorPrice  float32  `json:"slot_floor_price"`
	ImpressionValue float32  `json:"impression_value"`
	Timestamp       int64    `json:"timestamp"`
	ClearingPrice   float32  `json:"clearing_price"`
}

func New(cfg *config.Config, st *stats.Store, seed int64) (*Engine, error) {
	e := &Engine{
		cfg:  cfg,
		st:   st,
		pool: competitors.New(cfg.Competitors.Count, cfg.Competitors.BaseMedian, cfg.Competitors.Spread, seed),
		rng:  rand.New(rand.NewSource(seed)),
	}

	for _, c := range cfg.Companies {
		conn, err := grpc.NewClient(c.Address, grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			return nil, fmt.Errorf("dial %s: %w", c.Address, err)
		}
		bud := c.BudgetFor(cfg.Auction.DailyBudget)
		e.comps = append(e.comps, &company{
			id:          c.ID,
			client:      pb.NewBidServiceClient(conn),
			metricsURL:  metricsURL(c.Address, c.MetricsAddress),
			budget:      bud,
			dailyBudget: bud,
		})
		st.SetBudget(c.ID, bud)
	}

	if cfg.OutcomeLog != "" {
		f, err := os.OpenFile(cfg.OutcomeLog, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
		if err != nil {
			return nil, fmt.Errorf("open outcome log: %w", err)
		}
		e.logF = f
		e.logW = bufio.NewWriter(f)
	}
	return e, nil
}

// makeRequest builds a random but plausible impression to auction.
func (e *Engine) makeRequest() *pb.BidRequest {
	e.counter++
	size := slotSizes[e.rng.Intn(len(slotSizes))]
	floor := float32(0)
	if e.rng.Float64() < 0.5 {
		floor = float32(e.rng.Intn(100))
	}
	// a handful of random known tags
	var tags []string
	ntags := e.rng.Intn(4)
	for i := 0; i < ntags; i++ {
		tags = append(tags, tagPool[e.rng.Intn(len(tagPool))])
	}
	// timestamp: a day in the iPinYou range plus a random hour
	day := 6 + e.rng.Intn(7) // June 6..12
	hour := e.rng.Intn(24)
	ts := int64(20130600000000) + int64(day)*1000000 + int64(hour)*10000

	return &pb.BidRequest{
		RequestId:       fmt.Sprintf("auc-%d", e.counter),
		Region:          regions[e.rng.Intn(len(regions))],
		City:            cities[e.rng.Intn(len(cities))],
		Domain:          domains[e.rng.Intn(len(domains))],
		AdExchange:      int32(1 + e.rng.Intn(3)),
		SlotWidth:       size[0],
		SlotHeight:      size[1],
		SlotVisibility:  int32(e.rng.Intn(3)),
		SlotFormat:      int32(e.rng.Intn(2)),
		AdvertiserId:    advertisers[e.rng.Intn(len(advertisers))],
		UserTags:        tags,
		SlotFloorPrice:  floor,
		ImpressionValue: float32(e.cfg.Auction.ImpressionValue),
		Timestamp:       ts,
	}
}

// bidResult is one company's full answer to an auction: the bid plus everything
// the model and the transport told us (win prob, expected profit, the inference
// time the C++ server reported, and the round-trip time we measured).
type bidResult struct {
	comp      *company
	bid       float64
	winProb   float64
	expProfit float64
	infUs     float64
	rttUs     float64
	fallback  bool
	paced     bool
	ok        bool // false if it missed the deadline or errored
}

// slotKey is the "WxH" label used for the impression-mix breakdown.
func slotKey(w, h int32) string {
	return fmt.Sprintf("%dx%d", w, h)
}

// runAuction runs one full auction end to end.
func (e *Engine) runAuction(ctx context.Context) {
	req := e.makeRequest()
	deadline := time.Duration(e.cfg.Auction.DeadlineMs) * time.Millisecond

	// budget pacing: each company has a daily budget that resets every
	// day_auctions auctions. A company that has spent more than its even-paced
	// share by this point in the day sits out some auctions, so it does not burn
	// its whole budget early - which lets the other strategies win their share.
	day := e.cfg.Auction.DayAuctions
	if day > 0 {
		pos := (e.counter - 1) % int64(day)
		for _, c := range e.comps {
			// each company paces against its OWN daily budget, so an instance
			// given a smaller budget spends its share earlier and sits out more
			// of the day - genuinely budget-paced, not just a relabelled clone.
			c.mu.Lock()
			if pos == 0 {
				c.budget = c.dailyBudget
			}
			expectedSpent := c.dailyBudget * float64(pos) / float64(day)
			spent := c.dailyBudget - c.budget
			c.mu.Unlock()
			if expectedSpent > 0 && spent > expectedSpent {
				paceRatio := spent / expectedSpent // >1 = ahead of pace
				if e.rng.Float64() > 1.0/paceRatio {
					c.paced = true // sit this one out
					continue
				}
			}
			c.paced = false
		}
	}

	// ask every participating company for a bid at the same time, with the deadline
	results := make([]bidResult, len(e.comps))
	var wg sync.WaitGroup
	for i, c := range e.comps {
		if c.paced {
			results[i] = bidResult{comp: c, ok: false, paced: true}
			continue
		}
		wg.Add(1)
		go func(i int, c *company) {
			defer wg.Done()
			cctx, cancel := context.WithTimeout(ctx, deadline)
			defer cancel()
			start := time.Now()
			resp, err := c.client.GetBid(cctx, req)
			rtt := float64(time.Since(start).Microseconds())
			if err != nil {
				results[i] = bidResult{comp: c, ok: false, rttUs: rtt}
				return
			}
			results[i] = bidResult{
				comp:      c,
				bid:       float64(resp.BidPrice),
				winProb:   float64(resp.WinProbability),
				expProfit: float64(resp.ExpectedProfit),
				infUs:     float64(resp.InferenceUs),
				rttUs:     rtt,
				fallback:  resp.UsedFallback,
				ok:        true,
			}
		}(i, c)
	}
	wg.Wait()

	// find the highest bid among companies that can actually afford it
	var winner *company
	winningBid := 0.0
	for _, r := range results {
		if !r.ok || r.bid <= 0 {
			continue
		}
		r.comp.mu.Lock()
		afford := r.comp.budget >= r.bid
		r.comp.mu.Unlock()
		if afford && r.bid > winningBid {
			winningBid = r.bid
			winner = r.comp
		}
	}

	// the synthetic competitors also bid; the highest of everyone wins
	compBids := e.pool.Bids()
	competitorTop := 0.0
	statsComp := make([]stats.CompetitorBid, 0, len(compBids))
	for _, b := range compBids {
		if b.Amount > competitorTop {
			competitorTop = b.Amount
		}
		statsComp = append(statsComp, stats.CompetitorBid{Archetype: b.Archetype, Amount: b.Amount})
	}
	companyWon := winner != nil && winningBid >= competitorTop
	clearing := winningBid
	if competitorTop > clearing {
		clearing = competitorTop
	}

	V := e.cfg.Auction.ImpressionValue

	// settle: if a company had the top bid, it wins and pays its bid
	if companyWon {
		winner.mu.Lock()
		winner.budget -= winningBid
		winner.mu.Unlock()
	}

	// build the per-company telemetry entries and notify each bidder of the result
	entries := make([]stats.BidEntry, 0, len(results))
	for _, r := range results {
		won := companyWon && r.comp == winner
		profit := 0.0
		spend := 0.0
		if won {
			// first-price: the winner pays its own bid, so profit = V - bid. This
			// can be NEGATIVE: a strategy whose multiplier pushes its bid above V
			// (e.g. aggressive 1.2x) overpays and books a real loss on every win.
			// We surface that loss rather than flooring it - the scoreboard should
			// show an unprofitable strategy as unprofitable.
			profit = V - r.bid
			spend = r.bid
		}
		r.comp.mu.Lock()
		budget := r.comp.budget
		r.comp.mu.Unlock()
		entries = append(entries, stats.BidEntry{
			ID:        r.comp.id,
			OK:        r.ok,
			Paced:     r.paced,
			Won:       won,
			Bid:       r.bid,
			WinProb:   r.winProb,
			ExpProfit: r.expProfit,
			InfUs:     r.infUs,
			RttUs:     r.rttUs,
			Fallback:  r.fallback,
			Profit:    profit,
			Spend:     spend,
			Budget:    budget,
		})
		if r.ok {
			e.notify(ctx, r.comp, req.RequestId, won, float32(clearing), float32(r.bid), float32(profit))
		}
	}

	winnerID := ""
	if companyWon {
		winnerID = winner.id
	} else if competitorTop > 0 {
		winnerID = "market"
	}

	hour := int((req.Timestamp % 1000000) / 10000)
	e.st.Record(stats.Auction{
		RequestID:      req.RequestId,
		Hour:           hour,
		Exchange:       req.AdExchange,
		Slot:           slotKey(req.SlotWidth, req.SlotHeight),
		Domain:         req.Domain,
		Floor:          float64(req.SlotFloorPrice),
		HasFloor:       req.SlotFloorPrice > 0,
		Clearing:       clearing,
		CompetitorTop:  competitorTop,
		CompetitorBids: statsComp,
		Winner:         winnerID,
		V:              V,
		Entries:        entries,
	})
	e.writeLog(req, float32(clearing))
}

// notify tells a company the auction result (fire and forget, short timeout).
func (e *Engine) notify(ctx context.Context, c *company, reqID string, won bool, clearing, yourBid, profit float32) {
	cctx, cancel := context.WithTimeout(ctx, 50*time.Millisecond)
	defer cancel()
	_, _ = c.client.NotifyOutcome(cctx, &pb.AuctionOutcome{
		RequestId:     reqID,
		Won:           won,
		ClearingPrice: clearing,
		YourBid:       yourBid,
		Profit:        profit,
	})
}

func (e *Engine) writeLog(req *pb.BidRequest, clearing float32) {
	if e.logW == nil {
		return
	}
	rec := logRecord{
		RequestID:       req.RequestId,
		Region:          req.Region,
		City:            req.City,
		Domain:          req.Domain,
		AdExchange:      req.AdExchange,
		SlotWidth:       req.SlotWidth,
		SlotHeight:      req.SlotHeight,
		SlotVisibility:  req.SlotVisibility,
		SlotFormat:      req.SlotFormat,
		AdvertiserID:    req.AdvertiserId,
		UserTags:        req.UserTags,
		SlotFloorPrice:  req.SlotFloorPrice,
		ImpressionValue: req.ImpressionValue,
		Timestamp:       req.Timestamp,
		ClearingPrice:   clearing,
	}
	b, err := json.Marshal(&rec)
	if err != nil {
		return
	}
	e.logMu.Lock()
	e.logW.Write(b)
	e.logW.WriteByte('\n')
	e.logMu.Unlock()
}

// requestFromRecord rebuilds a BidRequest from a logged/replay record.
func (e *Engine) requestFromRecord(rec *logRecord) *pb.BidRequest {
	V := float32(e.cfg.Auction.ImpressionValue)
	if rec.ImpressionValue > 0 {
		V = rec.ImpressionValue
	}
	return &pb.BidRequest{
		RequestId:       rec.RequestID,
		Region:          rec.Region,
		City:            rec.City,
		Domain:          rec.Domain,
		AdExchange:      rec.AdExchange,
		SlotWidth:       rec.SlotWidth,
		SlotHeight:      rec.SlotHeight,
		SlotVisibility:  rec.SlotVisibility,
		SlotFormat:      rec.SlotFormat,
		AdvertiserId:    rec.AdvertiserID,
		UserTags:        rec.UserTags,
		SlotFloorPrice:  rec.SlotFloorPrice,
		ImpressionValue: V,
		Timestamp:       rec.Timestamp,
	}
}

// replayOne scores every company on a single recorded auction. Unlike a live
// auction (one winner), here each company is scored independently against the
// fixed clearing price - this is a backtest: "how would each strategy have done
// on this market?". No synthetic competitors, no budget.
func (e *Engine) replayOne(ctx context.Context, req *pb.BidRequest, clearing, V float64, deadline time.Duration) {
	entries := make([]stats.BidEntry, 0, len(e.comps))
	for _, c := range e.comps {
		cctx, cancel := context.WithTimeout(ctx, deadline)
		start := time.Now()
		resp, err := c.client.GetBid(cctx, req)
		rtt := float64(time.Since(start).Microseconds())
		cancel()
		if err != nil {
			entries = append(entries, stats.BidEntry{ID: c.id, OK: false, RttUs: rtt})
			continue
		}
		bid := float64(resp.BidPrice)
		won := bid > 0 && bid >= clearing
		profit := 0.0
		spend := 0.0
		if won {
			// first-price profit = V - bid, allowed to go negative (a bid above V
			// is a real loss); we do not floor it. See the live settle path.
			profit = V - bid
			spend = bid
		}
		entries = append(entries, stats.BidEntry{
			ID:        c.id,
			OK:        true,
			Won:       won,
			Bid:       bid,
			WinProb:   float64(resp.WinProbability),
			ExpProfit: float64(resp.ExpectedProfit),
			InfUs:     float64(resp.InferenceUs),
			RttUs:     rtt,
			Fallback:  resp.UsedFallback,
			Profit:    profit,
			Spend:     spend,
			Budget:    e.cfg.Auction.DailyBudget,
		})
	}
	hour := int((req.Timestamp % 1000000) / 10000)
	e.st.Record(stats.Auction{
		RequestID: req.RequestId,
		Hour:      hour,
		Exchange:  req.AdExchange,
		Slot:      slotKey(req.SlotWidth, req.SlotHeight),
		Domain:    req.Domain,
		Floor:     float64(req.SlotFloorPrice),
		HasFloor:  req.SlotFloorPrice > 0,
		Clearing:  clearing,
		Winner:    "backtest",
		V:         V,
		Entries:   entries,
	})
}

// RunReplay replays auctions from a JSONL file in the engine's own outcome-log
// format, instead of synthesizing them. Useful for deterministic regression runs
// and for backtesting strategies against a recorded market.
func (e *Engine) RunReplay(ctx context.Context, path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	V := e.cfg.Auction.ImpressionValue
	deadline := time.Duration(e.cfg.Auction.DeadlineMs) * time.Millisecond
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1024*1024), 1024*1024)
	count := 0
	for sc.Scan() {
		if ctx.Err() != nil {
			break
		}
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var rec logRecord
		if err := json.Unmarshal(line, &rec); err != nil {
			continue
		}
		req := e.requestFromRecord(&rec)
		e.replayOne(ctx, req, float64(rec.ClearingPrice), V, deadline)
		count++
	}
	return sc.Err()
}

// Run drives auctions until the context is cancelled. If rate_per_sec is set it
// paces auctions; otherwise it runs as fast as it can.
func (e *Engine) Run(ctx context.Context) {
	var tick <-chan time.Time
	if e.cfg.Auction.RatePerSec > 0 {
		interval := time.Second / time.Duration(e.cfg.Auction.RatePerSec)
		t := time.NewTicker(interval)
		defer t.Stop()
		tick = t.C
	}

	flush := time.NewTicker(1 * time.Second)
	defer flush.Stop()

	for {
		select {
		case <-ctx.Done():
			e.flushLog()
			return
		case <-flush.C:
			e.flushLog()
		default:
			if tick != nil {
				select {
				case <-ctx.Done():
					e.flushLog()
					return
				case <-tick:
				}
			}
			e.runAuction(ctx)
		}
	}
}

func (e *Engine) flushLog() {
	if e.logW == nil {
		return
	}
	e.logMu.Lock()
	e.logW.Flush()
	e.logMu.Unlock()
}

// Close flushes and closes the outcome log.
func (e *Engine) Close() {
	e.flushLog()
	if e.logF != nil {
		e.logF.Close()
	}
}
