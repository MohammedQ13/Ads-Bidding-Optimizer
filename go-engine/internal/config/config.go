// Package config loads the engine settings from a YAML file. Like the C++ side,
// every field has a sensible default so a missing or partial config still runs.
package config

import (
	"os"
	"strconv"

	"gopkg.in/yaml.v3"
)

// Company is one C++ bid server we send auctions to. MetricsAddress is the
// host:port of that server's Prometheus /metrics page; if empty the engine
// derives it from Address (same host, port 9100) so it can scrape the C++
// internals (circuit breaker, queue depth, cache hits, model version).
type Company struct {
	ID             string `yaml:"id"`
	Address        string `yaml:"address"`
	MetricsAddress string `yaml:"metrics_address"`
	// Budget overrides the shared auction.daily_budget for this one instance.
	// 0 means "use the shared budget". A smaller budget makes a strategy genuinely
	// budget-paced: it spends its share earlier and sits out the rest of the day,
	// so it wins fewer auctions than an identically-bidding instance.
	Budget float64 `yaml:"budget"`
}

// BudgetFor returns the company's per-instance budget, falling back to the shared
// daily budget when no override is set.
func (c Company) BudgetFor(shared float64) float64 {
	if c.Budget > 0 {
		return c.Budget
	}
	return shared
}

// Config holds everything the engine needs to run.
type Config struct {
	Companies []Company `yaml:"companies"`

	Auction struct {
		RatePerSec      int     `yaml:"rate_per_sec"`     // auctions/sec, 0 = flat out
		DeadlineMs      int     `yaml:"deadline_ms"`      // bid deadline, late ones excluded
		ImpressionValue float64 `yaml:"impression_value"` // V per auction
		DailyBudget     float64 `yaml:"daily_budget"`     // per-company budget in fen
		DayAuctions     int     `yaml:"day_auctions"`     // auctions per simulated day (budget resets)
	} `yaml:"auction"`

	Competitors struct {
		Count      int     `yaml:"count"`       // synthetic competitors per auction
		BaseMedian float64 `yaml:"base_median"` // median market price in fen
		Spread     float64 `yaml:"spread"`      // lognormal sigma
	} `yaml:"competitors"`

	MetricsPort     int    `yaml:"metrics_port"`
	OutcomeLog      string `yaml:"outcome_log"`      // where to append auction outcomes
	RetrainerStatus string `yaml:"retrainer_status"` // JSON the retrainer writes each round
	ScrapeMetrics   bool   `yaml:"scrape_metrics"`   // poll each company's /metrics page
}

// Load reads the YAML file and fills in defaults for anything missing.
func Load(path string) (*Config, error) {
	c := &Config{}
	// defaults first
	c.Auction.RatePerSec = 200
	c.Auction.DeadlineMs = 10
	c.Auction.ImpressionValue = 150
	c.Auction.DailyBudget = 200000
	c.Auction.DayAuctions = 2000
	c.Competitors.Count = 4
	c.Competitors.BaseMedian = 75
	c.Competitors.Spread = 0.6
	c.MetricsPort = 9200
	c.OutcomeLog = "/data/outcomes.jsonl"
	c.RetrainerStatus = "/models/retrainer_status.json"
	c.ScrapeMetrics = true

	data, err := os.ReadFile(path)
	if err != nil {
		// no file is fine, just run on defaults (but there are no companies then)
		applyEnvOverrides(c)
		return c, nil
	}
	if err := yaml.Unmarshal(data, c); err != nil {
		return nil, err
	}
	applyEnvOverrides(c)
	return c, nil
}

// applyEnvOverrides lets you tune the run without editing the YAML, which makes
// it easy to push the system hard from a compose file or a single command:
//
//	RATE_PER_SEC=0   -> run auctions flat out (0 = as fast as the loop allows)
//	COMPETITORS_COUNT=16, DEADLINE_MS=20, IMPRESSION_VALUE=200
func applyEnvOverrides(c *Config) {
	if v, ok := envInt("RATE_PER_SEC"); ok {
		c.Auction.RatePerSec = v
	}
	if v, ok := envInt("DEADLINE_MS"); ok {
		c.Auction.DeadlineMs = v
	}
	if v, ok := envFloat("IMPRESSION_VALUE"); ok {
		c.Auction.ImpressionValue = v
	}
	if v, ok := envInt("COMPETITORS_COUNT"); ok {
		c.Competitors.Count = v
	}
	if v, ok := envFloat("COMPETITORS_BASE_MEDIAN"); ok {
		c.Competitors.BaseMedian = v
	}
}

func envInt(key string) (int, bool) {
	s := os.Getenv(key)
	if s == "" {
		return 0, false
	}
	n, err := strconv.Atoi(s)
	if err != nil {
		return 0, false
	}
	return n, true
}

func envFloat(key string) (float64, bool) {
	s := os.Getenv(key)
	if s == "" {
		return 0, false
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, false
	}
	return f, true
}
