// Scraping the C++ servers and watching the retrainer. The Go engine is the one
// process that talks to everything, so it doubles as the telemetry collector: it
// periodically reads each company's Prometheus /metrics page for the internals
// the bid response cannot carry (circuit breaker state, batch queue depth, cache
// hits, loaded model version, cold-start lookups), and it reads the JSON status
// the retrainer writes each round. Both feed the live /stats the dashboard polls.
package engine

import (
	"bufio"
	"context"
	"encoding/json"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"rtbengine/internal/stats"
)

// metricsURL works out where a company's /metrics page is. If the config gives
// an explicit metrics_address we use it; otherwise we assume the same host as
// the gRPC address with the standard exporter port 9100.
func metricsURL(grpcAddr, override string) string {
	addr := override
	if addr == "" {
		host := grpcAddr
		if i := strings.LastIndex(grpcAddr, ":"); i >= 0 {
			host = grpcAddr[:i]
		}
		addr = host + ":9100"
	}
	return "http://" + addr + "/metrics"
}

// StartCollectors launches the background scrapers. They run until ctx is done.
func (e *Engine) StartCollectors(ctx context.Context) {
	if e.cfg.ScrapeMetrics {
		go e.scrapeLoop(ctx)
	}
	if e.cfg.RetrainerStatus != "" {
		go e.retrainerLoop(ctx)
	}
}

func (e *Engine) scrapeLoop(ctx context.Context) {
	client := &http.Client{Timeout: 1500 * time.Millisecond}
	t := time.NewTicker(2 * time.Second)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			for _, c := range e.comps {
				res := scrapeOne(client, c.metricsURL)
				e.st.SetScrape(c.id, res)
			}
		}
	}
}

// scrapeOne pulls one /metrics page and pulls out the gauges/counters we show.
func scrapeOne(client *http.Client, url string) stats.ScrapeResult {
	res := stats.ScrapeResult{}
	resp, err := client.Get(url)
	if err != nil {
		return res
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return res
	}

	// Prometheus text format: "metric_name{labels} value". We only need a handful
	// of single-series metrics, so a line scan is plenty - no full parser needed.
	var batchSum, batchCount, reqTotal float64
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || line[0] == '#' {
			continue
		}
		name, val, ok := splitMetric(line)
		if !ok {
			continue
		}
		switch {
		// total requests served, summed across all status labels - this counts
		// EVERY gRPC request the server handled (incl. load-generator traffic), so
		// its rate is the real serving throughput, not just the demo exchange's.
		case strings.HasPrefix(name, "bid_request_total"):
			reqTotal += val
		case strings.HasPrefix(name, "bid_circuit_breaker_state"):
			res.BreakerState = int(val)
		case strings.HasPrefix(name, "bid_request_queue_depth"):
			res.QueueDepth = int(val)
		case strings.HasPrefix(name, "bid_model_version"):
			res.ModelVersion = val
		case strings.HasPrefix(name, "bid_cache_hit_total"):
			res.CacheHits = int64(val)
		case strings.HasPrefix(name, "bid_cold_start_lookup_total"):
			res.ColdStart = int64(val)
		case strings.HasPrefix(name, "bid_batch_size_sum"):
			batchSum = val
		case strings.HasPrefix(name, "bid_batch_size_count"):
			batchCount = val
		}
	}
	// pass the raw histogram sum/count through; the store turns them into a RECENT
	// batch size from the delta between scrapes, so it reflects batching happening
	// now (≈1 at rest, higher under load) rather than a sticky lifetime average.
	res.BatchSum = batchSum
	res.BatchCount = batchCount
	res.RequestTotal = int64(reqTotal)
	res.OK = true
	return res
}

// splitMetric pulls the metric name and float value out of one exposition line.
func splitMetric(line string) (string, float64, bool) {
	sp := strings.LastIndex(line, " ")
	if sp < 0 {
		return "", 0, false
	}
	name := line[:sp]
	v, err := strconv.ParseFloat(strings.TrimSpace(line[sp+1:]), 64)
	if err != nil {
		return "", 0, false
	}
	return name, v, true
}

// retrainerStatus mirrors the JSON the Python retrainer writes each round.
type retrainerStatus struct {
	Round        int     `json:"round"`
	Rows         int     `json:"rows"`
	LastLoss     float64 `json:"last_loss"`
	ExportedAt   int64   `json:"exported_at"`
	ModelVersion float64 `json:"model_version"`
	Window       int     `json:"window"`
	Exports      int     `json:"exports"`
	Status       string  `json:"status"`
}

func (e *Engine) retrainerLoop(ctx context.Context) {
	t := time.NewTicker(3 * time.Second)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			e.readRetrainer()
		}
	}
}

func (e *Engine) readRetrainer() {
	data, err := os.ReadFile(e.cfg.RetrainerStatus)
	if err != nil {
		e.st.SetRetrainer(stats.RetrainerOut{Available: false})
		return
	}
	var rs retrainerStatus
	if err := json.Unmarshal(data, &rs); err != nil {
		e.st.SetRetrainer(stats.RetrainerOut{Available: false})
		return
	}
	e.st.SetRetrainer(stats.RetrainerOut{
		Available:    true,
		Round:        rs.Round,
		Rows:         rs.Rows,
		LastLoss:     rs.LastLoss,
		ExportedAt:   rs.ExportedAt,
		ModelVersion: rs.ModelVersion,
		Window:       rs.Window,
		Exports:      rs.Exports,
		Status:       rs.Status,
	})
}
