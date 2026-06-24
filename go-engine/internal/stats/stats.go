// Package stats holds the live telemetry for the whole RTB system and serves it
// over plain HTTP as JSON (/stats) and Server-Sent Events (/events). It is the
// single source of truth the custom dashboard reads, so it deliberately exposes
// a lot: per-strategy economics, the model's predictions, end-to-end and
// inference latency percentiles, the synthetic market, a live auction feed, the
// C++ servers' internal counters (scraped from their /metrics), and the state of
// the retraining loop. This replaces Prometheus/Grafana with one rich endpoint.
package stats

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"sync"
	"time"
)

// keep these bounded so memory and JSON size stay small
const (
	latSamples = 512 // recent latency samples kept per company for percentiles
	eventRing  = 80  // recent auctions kept for the live feed
	apsRing    = 256 // recent auction timestamps kept for throughput
	lossRing   = 60  // recent retrain losses kept for the loss curve
)

// price histogram buckets in fen: 0-10, 10-20, ... 190-200, 200+
var priceEdges = []float64{0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
	110, 120, 130, 140, 150, 160, 170, 180, 190, 200}

// ---------- per-company running state (internal) ----------

type compState struct {
	// economics
	won, lost, noBid int64
	profit, spend    float64
	budget           float64
	// model + bids
	bids         int64
	bidSum       float64
	lastBid      float64
	winProbSum   float64
	expProfitSum float64
	fallback     int64
	// latency rings (microseconds)
	rtt    []float64
	inf    []float64
	rttPos int
	infPos int
	// scraped C++ internals (0 if not scraped)
	breakerState int
	queueDepth   int
	modelVersion float64
	cacheHits    int64
	coldStart    int64
	// batchAvg is the RECENT avg batch size, from the sum/count counter deltas
	// between scrapes (≈1 at rest, higher when the batcher coalesces under load).
	batchAvg      float64
	batchSumPrev  float64
	batchCntPrev  float64
	batchHavePrev bool
	scraped       bool
	healthy       bool
	lastSeen      time.Time
	// real serving throughput, derived from the bid_request_total counter delta
	// between scrapes. This is the true req/s the C++ server handles (including any
	// load-generator traffic), as opposed to the demo exchange's own auction rate.
	reqTotal     int64
	reqPrev      int64
	reqPrevTime  time.Time
	servedPerSec float64
}

func newCompState(budget float64) *compState {
	return &compState{budget: budget, healthy: true}
}

func (c *compState) pushRtt(us float64) {
	if len(c.rtt) < latSamples {
		c.rtt = append(c.rtt, us)
		return
	}
	c.rtt[c.rttPos] = us
	c.rttPos = (c.rttPos + 1) % latSamples
}

func (c *compState) pushInf(us float64) {
	if len(c.inf) < latSamples {
		c.inf = append(c.inf, us)
		return
	}
	c.inf[c.infPos] = us
	c.infPos = (c.infPos + 1) % latSamples
}

// ---------- inputs the engine hands us ----------

// BidEntry is one company's answer to one auction, with everything the model and
// the transport told us about it.
type BidEntry struct {
	ID        string
	OK        bool // a real bid came back before the deadline
	Paced     bool // sat out this auction for budget pacing
	Won       bool
	Bid       float64
	WinProb   float64
	ExpProfit float64
	InfUs     float64 // inference time the C++ server reported
	RttUs     float64 // round-trip time the engine measured
	Fallback  bool
	Profit    float64
	Spend     float64
	Budget    float64
}

// CompetitorBid is one synthetic competitor's bid plus its archetype label.
type CompetitorBid struct {
	Archetype string
	Amount    float64
}

// ScrapeResult is what we pull from a C++ server's Prometheus /metrics page.
type ScrapeResult struct {
	OK           bool
	BreakerState int
	QueueDepth   int
	ModelVersion float64
	CacheHits    int64
	ColdStart    int64
	BatchSum     float64 // cumulative bid_batch_size_sum (requests batched)
	BatchCount   float64 // cumulative bid_batch_size_count (batches processed)
	RequestTotal int64   // cumulative bid_request_total (summed across statuses)
}

// Auction is the full record of one auction the engine just ran.
type Auction struct {
	RequestID      string
	Hour           int
	Exchange       int32
	Slot           string
	Domain         string
	Floor          float64
	HasFloor       bool
	Weekend        bool
	Clearing       float64
	CompetitorTop  float64
	CompetitorBids []CompetitorBid
	Winner         string // company id, or "" / "market" if competitors won
	V              float64
	Entries        []BidEntry
}

// ---------- the live store ----------

type Store struct {
	mu        sync.Mutex
	startedAt time.Time

	auctions     int64
	lastClearing float64
	impValue     float64
	deadlineMs   int
	numComp      int

	companies map[string]*compState
	order     []string // stable company order (insertion order)

	// market histograms
	clearingHist []int64
	bidHist      []int64
	compHist     []int64
	winProbHist  []int64 // 20 buckets of 0.05

	// impression mix
	byHour     [24]int64
	byExchange map[int32]int64
	bySlot     map[string]int64
	floorSeen  int64

	// competitor archetype tallies
	archCount map[string]int64
	archSum   map[string]float64
	archLast  map[string]float64

	// rolling state
	apsTimes []time.Time
	apsPos   int
	events   []EventOut // ring of recent auctions, newest last
	evPos    int

	// retrainer status (read from disk)
	retr     RetrainerOut
	lossHist []float64
}

func New() *Store {
	return &Store{
		startedAt:    time.Now(),
		companies:    make(map[string]*compState),
		clearingHist: make([]int64, len(priceEdges)),
		bidHist:      make([]int64, len(priceEdges)),
		compHist:     make([]int64, len(priceEdges)),
		winProbHist:  make([]int64, 20),
		byExchange:   make(map[int32]int64),
		bySlot:       make(map[string]int64),
		archCount:    make(map[string]int64),
		archSum:      make(map[string]float64),
		archLast:     make(map[string]float64),
	}
}

// Configure records a few engine-wide settings so the dashboard can show them.
func (s *Store) Configure(impValue float64, deadlineMs, numCompetitors int) {
	s.mu.Lock()
	s.impValue = impValue
	s.deadlineMs = deadlineMs
	s.numComp = numCompetitors
	s.mu.Unlock()
}

func (s *Store) company(id string) *compState {
	c := s.companies[id]
	if c == nil {
		c = newCompState(0)
		s.companies[id] = c
		s.order = append(s.order, id)
	}
	return c
}

// SetBudget seeds a company's starting budget (called at startup).
func (s *Store) SetBudget(id string, budget float64) {
	s.mu.Lock()
	s.company(id).budget = budget
	s.mu.Unlock()
}

// priceBucket maps a fen price to its histogram index.
func priceBucket(v float64) int {
	idx := int(v / 10)
	if idx < 0 {
		idx = 0
	}
	if idx >= len(priceEdges) {
		idx = len(priceEdges) - 1
	}
	return idx
}

// Record folds one finished auction into all the running telemetry.
func (s *Store) Record(a Auction) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.auctions++
	s.lastClearing = a.Clearing

	// throughput ring
	now := time.Now()
	if len(s.apsTimes) < apsRing {
		s.apsTimes = append(s.apsTimes, now)
	} else {
		s.apsTimes[s.apsPos] = now
		s.apsPos = (s.apsPos + 1) % apsRing
	}

	// impression mix
	if a.Hour >= 0 && a.Hour < 24 {
		s.byHour[a.Hour]++
	}
	s.byExchange[a.Exchange]++
	s.bySlot[a.Slot]++
	if a.HasFloor {
		s.floorSeen++
	}

	// clearing price histogram
	s.clearingHist[priceBucket(a.Clearing)]++

	// competitor distribution + archetypes
	for _, cb := range a.CompetitorBids {
		s.compHist[priceBucket(cb.Amount)]++
		s.archCount[cb.Archetype]++
		s.archSum[cb.Archetype] += cb.Amount
		s.archLast[cb.Archetype] = cb.Amount
	}

	// per-company updates
	for _, e := range a.Entries {
		c := s.company(e.ID)
		c.lastSeen = now
		if e.Paced {
			// paced companies did not bid; not a failure, just sat out
			continue
		}
		if !e.OK {
			c.noBid++
			c.healthy = false
			continue
		}
		c.healthy = true
		c.bids++
		c.bidSum += e.Bid
		c.lastBid = e.Bid
		c.winProbSum += e.WinProb
		c.expProfitSum += e.ExpProfit
		if e.Fallback {
			c.fallback++
		}
		if e.RttUs > 0 {
			c.pushRtt(e.RttUs)
		}
		if e.InfUs > 0 {
			c.pushInf(e.InfUs)
		}
		// bid + win-prob histograms
		s.bidHist[priceBucket(e.Bid)]++
		wpIdx := int(e.WinProb * 20)
		if wpIdx < 0 {
			wpIdx = 0
		}
		if wpIdx >= 20 {
			wpIdx = 19
		}
		s.winProbHist[wpIdx]++
		// outcome
		if e.Won {
			c.won++
			c.profit += e.Profit
			c.spend += e.Spend
			c.budget = e.Budget
		} else {
			c.lost++
		}
	}

	// live event feed (trimmed copy)
	s.pushEvent(a, now)
}

// pushEvent appends a compact record of the auction to the ring (caller holds lock).
func (s *Store) pushEvent(a Auction, now time.Time) {
	ev := EventOut{
		ID:       a.RequestID,
		T:        now.UnixMilli(),
		Hour:     a.Hour,
		Exchange: a.Exchange,
		Slot:     a.Slot,
		Domain:   a.Domain,
		Floor:    round1(a.Floor),
		Clearing: round1(a.Clearing),
		CompTop:  round1(a.CompetitorTop),
		Winner:   a.Winner,
		Bids:     make([]EventBid, 0, len(a.Entries)),
	}
	for _, e := range a.Entries {
		ev.Bids = append(ev.Bids, EventBid{
			ID:        e.ID,
			OK:        e.OK,
			Paced:     e.Paced,
			Won:       e.Won,
			Bid:       round1(e.Bid),
			WinProb:   round3(e.WinProb),
			ExpProfit: round1(e.ExpProfit),
			InfUs:     int64(e.InfUs),
			RttUs:     int64(e.RttUs),
			Fallback:  e.Fallback,
		})
	}
	if len(s.events) < eventRing {
		s.events = append(s.events, ev)
	} else {
		s.events[s.evPos] = ev
		s.evPos = (s.evPos + 1) % eventRing
	}
}

// SetScrape stores the C++ internals scraped from a company's /metrics page.
func (s *Store) SetScrape(id string, m ScrapeResult) {
	s.mu.Lock()
	c := s.company(id)
	c.breakerState = m.BreakerState
	c.queueDepth = m.QueueDepth
	c.modelVersion = m.ModelVersion
	c.cacheHits = m.CacheHits
	c.coldStart = m.ColdStart
	// recent batch size from the counter deltas; guard the first sample and a
	// counter reset (server restart) by falling back to the cumulative average.
	if m.OK {
		if c.batchHavePrev && m.BatchCount > c.batchCntPrev && m.BatchSum >= c.batchSumPrev {
			c.batchAvg = (m.BatchSum - c.batchSumPrev) / (m.BatchCount - c.batchCntPrev)
		} else if m.BatchCount > 0 {
			c.batchAvg = m.BatchSum / m.BatchCount
		}
		c.batchSumPrev = m.BatchSum
		c.batchCntPrev = m.BatchCount
		c.batchHavePrev = true
	}
	c.scraped = m.OK
	// derive serving throughput from the request-counter delta between scrapes.
	// guard against the counter resetting (server restart) by only counting forward
	// deltas, and ignore the first sample (no baseline yet).
	if m.OK {
		now := time.Now()
		if !c.reqPrevTime.IsZero() && m.RequestTotal >= c.reqPrev {
			dt := now.Sub(c.reqPrevTime).Seconds()
			if dt > 0 {
				c.servedPerSec = round1(float64(m.RequestTotal-c.reqPrev) / dt)
			}
		}
		c.reqPrev = m.RequestTotal
		c.reqPrevTime = now
		c.reqTotal = m.RequestTotal
	}
	s.mu.Unlock()
}

// SetRetrainer stores the latest retrainer status read from disk.
func (s *Store) SetRetrainer(r RetrainerOut) {
	s.mu.Lock()
	prevRound := s.retr.Round
	s.retr = r
	// keep a short loss curve, one point per new round
	if r.Available && r.Round != prevRound {
		s.lossHist = append(s.lossHist, r.LastLoss)
		if len(s.lossHist) > lossRing {
			s.lossHist = s.lossHist[len(s.lossHist)-lossRing:]
		}
	}
	s.mu.Unlock()
}

// ---------- output (JSON) shapes ----------

type LatencyOut struct {
	P50 float64 `json:"p50"`
	P95 float64 `json:"p95"`
	P99 float64 `json:"p99"`
	Avg float64 `json:"avg"`
}

type CompanyOut struct {
	Won          int64      `json:"won"`
	Lost         int64      `json:"lost"`
	NoBid        int64      `json:"no_bid"`
	Profit       float64    `json:"profit"`
	Spend        float64    `json:"spend"`
	Budget       float64    `json:"budget"`
	WinRate      float64    `json:"win_rate"`
	ROI          float64    `json:"roi"`
	Bids         int64      `json:"bids"`
	AvgBid       float64    `json:"avg_bid"`
	LastBid      float64    `json:"last_bid"`
	AvgWinProb   float64    `json:"avg_win_prob"`
	ExpProfit    float64    `json:"expected_profit"`
	Fallback     int64      `json:"fallback"`
	FallbackRate float64    `json:"fallback_rate"`
	RTT          LatencyOut `json:"rtt_us"`
	Inference    LatencyOut `json:"inference_us"`
	BreakerState int        `json:"breaker_state"`
	QueueDepth   int        `json:"queue_depth"`
	ModelVersion float64    `json:"model_version"`
	CacheHits    int64      `json:"cache_hits"`
	ColdStart    int64      `json:"cold_start"`
	BatchAvg     float64    `json:"batch_avg"`
	ServedPerSec float64    `json:"served_per_sec"` // real req/s this server handles
	Healthy      bool       `json:"healthy"`
}

type Bucket struct {
	Lo    float64 `json:"lo"`
	Hi    float64 `json:"hi"`
	Count int64   `json:"count"`
}

type ArchetypeOut struct {
	Label   string  `json:"label"`
	Count   int64   `json:"count"`
	AvgBid  float64 `json:"avg_bid"`
	LastBid float64 `json:"last_bid"`
}

type EventBid struct {
	ID        string  `json:"id"`
	OK        bool    `json:"ok"`
	Paced     bool    `json:"paced"`
	Won       bool    `json:"won"`
	Bid       float64 `json:"bid"`
	WinProb   float64 `json:"win_prob"`
	ExpProfit float64 `json:"exp_profit"`
	InfUs     int64   `json:"inf_us"`
	RttUs     int64   `json:"rtt_us"`
	Fallback  bool    `json:"fallback"`
}

type EventOut struct {
	ID       string     `json:"id"`
	T        int64      `json:"t"`
	Hour     int        `json:"hour"`
	Exchange int32      `json:"exchange"`
	Slot     string     `json:"slot"`
	Domain   string     `json:"domain"`
	Floor    float64    `json:"floor"`
	Clearing float64    `json:"clearing"`
	CompTop  float64    `json:"competitor_top"`
	Winner   string     `json:"winner"`
	Bids     []EventBid `json:"bids"`
}

type RetrainerOut struct {
	Available    bool      `json:"available"`
	Round        int       `json:"round"`
	Rows         int       `json:"rows"`
	LastLoss     float64   `json:"last_loss"`
	ExportedAt   int64     `json:"exported_at"`
	ModelVersion float64   `json:"model_version"`
	Window       int       `json:"window"`
	Exports      int       `json:"exports"`
	Status       string    `json:"status"`
	LossHistory  []float64 `json:"loss_history,omitempty"`
}

type Snapshot struct {
	StartedAt       int64                 `json:"started_at"`
	UptimeS         float64               `json:"uptime_s"`
	Auctions        int64                 `json:"auctions"`
	AuctionsPerSec  float64               `json:"auctions_per_sec"`
	LastClearing    float64               `json:"last_clearing_price"`
	ImpressionValue float64               `json:"impression_value"`
	DeadlineMs      int                   `json:"deadline_ms"`
	NumCompetitors  int                   `json:"num_competitors"`
	Companies       map[string]CompanyOut `json:"companies"`
	Order           []string              `json:"order"`
	Histograms      map[string][]Bucket   `json:"histograms"`
	WinProbHist     []Bucket              `json:"win_prob_hist"`
	Impressions     ImpressionsOut        `json:"impressions"`
	Archetypes      []ArchetypeOut        `json:"competitor_archetypes"`
	Events          []EventOut            `json:"events"`
	Retrainer       RetrainerOut          `json:"retrainer"`
}

type ImpressionsOut struct {
	ByHour     []int64          `json:"by_hour"`
	ByExchange map[string]int64 `json:"by_exchange"`
	BySlot     map[string]int64 `json:"by_slot"`
	FloorRate  float64          `json:"floor_rate"`
}

// ---------- snapshot building ----------

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(p * float64(len(sorted)-1))
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

func latencyOut(samples []float64) LatencyOut {
	if len(samples) == 0 {
		return LatencyOut{}
	}
	cp := make([]float64, len(samples))
	copy(cp, samples)
	sort.Float64s(cp)
	var sum float64
	for _, v := range cp {
		sum += v
	}
	return LatencyOut{
		P50: round1(percentile(cp, 0.50)),
		P95: round1(percentile(cp, 0.95)),
		P99: round1(percentile(cp, 0.99)),
		Avg: round1(sum / float64(len(cp))),
	}
}

func histOut(counts []int64) []Bucket {
	out := make([]Bucket, len(counts))
	for i, n := range counts {
		lo := priceEdges[i]
		hi := lo + 10
		if i == len(counts)-1 {
			hi = 999 // open top bucket
		}
		out[i] = Bucket{Lo: lo, Hi: hi, Count: n}
	}
	return out
}

// Snapshot returns a copy of the whole telemetry tree, safe to serialize.
func (s *Store) Snapshot() Snapshot {
	s.mu.Lock()
	defer s.mu.Unlock()

	out := Snapshot{
		StartedAt:       s.startedAt.Unix(),
		UptimeS:         round1(time.Since(s.startedAt).Seconds()),
		Auctions:        s.auctions,
		AuctionsPerSec:  s.throughput(),
		LastClearing:    round1(s.lastClearing),
		ImpressionValue: s.impValue,
		DeadlineMs:      s.deadlineMs,
		NumCompetitors:  s.numComp,
		Companies:       make(map[string]CompanyOut, len(s.companies)),
		Order:           append([]string(nil), s.order...),
	}

	for id, c := range s.companies {
		total := c.won + c.lost
		wr := 0.0
		if total > 0 {
			wr = float64(c.won) / float64(total)
		}
		avgBid := 0.0
		avgWP := 0.0
		if c.bids > 0 {
			avgBid = c.bidSum / float64(c.bids)
			avgWP = c.winProbSum / float64(c.bids)
		}
		roi := 0.0
		if c.spend > 0 {
			roi = c.profit / c.spend
		}
		fbRate := 0.0
		if c.bids > 0 {
			fbRate = float64(c.fallback) / float64(c.bids)
		}
		out.Companies[id] = CompanyOut{
			Won:          c.won,
			Lost:         c.lost,
			NoBid:        c.noBid,
			Profit:       round1(c.profit),
			Spend:        round1(c.spend),
			Budget:       round1(c.budget),
			WinRate:      round3(wr),
			ROI:          round3(roi),
			Bids:         c.bids,
			AvgBid:       round1(avgBid),
			LastBid:      round1(c.lastBid),
			AvgWinProb:   round3(avgWP),
			ExpProfit:    round1(c.expProfitSum),
			Fallback:     c.fallback,
			FallbackRate: round3(fbRate),
			RTT:          latencyOut(c.rtt),
			Inference:    latencyOut(c.inf),
			BreakerState: c.breakerState,
			QueueDepth:   c.queueDepth,
			ModelVersion: c.modelVersion,
			CacheHits:    c.cacheHits,
			ColdStart:    c.coldStart,
			BatchAvg:     round2(c.batchAvg),
			ServedPerSec: c.servedPerSec,
			Healthy:      c.healthy,
		}
	}

	out.Histograms = map[string][]Bucket{
		"clearing":    histOut(s.clearingHist),
		"bids":        histOut(s.bidHist),
		"competitors": histOut(s.compHist),
	}
	out.WinProbHist = winProbBuckets(s.winProbHist)

	byHour := make([]int64, 24)
	copy(byHour, s.byHour[:])
	byExch := make(map[string]int64, len(s.byExchange))
	for k, v := range s.byExchange {
		byExch[fmt.Sprintf("%d", k)] = v
	}
	bySlot := make(map[string]int64, len(s.bySlot))
	for k, v := range s.bySlot {
		bySlot[k] = v
	}
	floorRate := 0.0
	if s.auctions > 0 {
		floorRate = float64(s.floorSeen) / float64(s.auctions)
	}
	out.Impressions = ImpressionsOut{
		ByHour: byHour, ByExchange: byExch, BySlot: bySlot, FloorRate: round3(floorRate),
	}

	arch := make([]ArchetypeOut, 0, len(s.archCount))
	for label, n := range s.archCount {
		avg := 0.0
		if n > 0 {
			avg = s.archSum[label] / float64(n)
		}
		arch = append(arch, ArchetypeOut{
			Label: label, Count: n, AvgBid: round1(avg), LastBid: round1(s.archLast[label]),
		})
	}
	sort.Slice(arch, func(i, j int) bool { return arch[i].AvgBid > arch[j].AvgBid })
	out.Archetypes = arch

	out.Events = s.eventsNewestFirst()

	r := s.retr
	if len(s.lossHist) > 0 {
		r.LossHistory = append([]float64(nil), s.lossHist...)
	}
	out.Retrainer = r

	return out
}

// throughput estimates auctions/sec from the recent-timestamp ring (caller holds lock).
func (s *Store) throughput() float64 {
	if len(s.apsTimes) < 2 {
		return 0
	}
	var oldest time.Time
	first := true
	for _, t := range s.apsTimes {
		if first || t.Before(oldest) {
			oldest = t
			first = false
		}
	}
	span := time.Since(oldest).Seconds()
	if span <= 0 {
		return 0
	}
	return round1(float64(len(s.apsTimes)-1) / span)
}

// eventsNewestFirst returns the event ring ordered newest first (caller holds lock).
func (s *Store) eventsNewestFirst() []EventOut {
	n := len(s.events)
	out := make([]EventOut, 0, n)
	if n < eventRing {
		for i := n - 1; i >= 0; i-- {
			out = append(out, s.events[i])
		}
		return out
	}
	// ring is full: evPos is the oldest slot
	for i := 0; i < n; i++ {
		idx := (s.evPos - 1 - i + n*2) % n
		out = append(out, s.events[idx])
	}
	return out
}

func winProbBuckets(counts []int64) []Bucket {
	out := make([]Bucket, len(counts))
	for i, n := range counts {
		out[i] = Bucket{Lo: float64(i) * 0.05, Hi: float64(i+1) * 0.05, Count: n}
	}
	return out
}

// ---------- HTTP ----------

// Serve starts the HTTP endpoints in the background. /stats is a JSON snapshot,
// /events is an SSE stream that pushes the snapshot about twice a second.
func (s *Store) Serve(port int) {
	mux := http.NewServeMux()
	mux.HandleFunc("/stats", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Access-Control-Allow-Origin", "*")
		json.NewEncoder(w).Encode(s.Snapshot())
	})
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Write([]byte("ok"))
	})
	mux.HandleFunc("/events", func(w http.ResponseWriter, r *http.Request) {
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming unsupported", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("Access-Control-Allow-Origin", "*")
		ticker := time.NewTicker(500 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-r.Context().Done():
				return
			case <-ticker.C:
				b, _ := json.Marshal(s.Snapshot())
				fmt.Fprintf(w, "data: %s\n\n", b)
				flusher.Flush()
			}
		}
	})
	addr := fmt.Sprintf("0.0.0.0:%d", port)
	go func() {
		// best effort: if the port is taken the engine still runs auctions
		_ = http.ListenAndServe(addr, mux)
	}()
}

// ---------- small rounding helpers (keep JSON tidy) ----------

func round1(v float64) float64 { return float64(int64(v*10+sign(v)*0.5)) / 10 }
func round2(v float64) float64 { return float64(int64(v*100+sign(v)*0.5)) / 100 }
func round3(v float64) float64 { return float64(int64(v*1000+sign(v)*0.5)) / 1000 }

func sign(v float64) float64 {
	if v < 0 {
		return -1
	}
	return 1
}
