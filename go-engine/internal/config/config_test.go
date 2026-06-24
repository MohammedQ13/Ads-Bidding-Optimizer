package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDefaultsWhenFileMissing(t *testing.T) {
	c, err := Load("/no/such/file.yaml")
	if err != nil {
		t.Fatalf("missing file should not error: %v", err)
	}
	if c.Auction.RatePerSec != 200 {
		t.Errorf("default rate_per_sec wrong: %d", c.Auction.RatePerSec)
	}
	if c.Auction.DeadlineMs != 10 {
		t.Errorf("default deadline_ms wrong: %d", c.Auction.DeadlineMs)
	}
	if c.Auction.ImpressionValue != 150 {
		t.Errorf("default impression_value wrong: %f", c.Auction.ImpressionValue)
	}
	if c.Competitors.Count != 4 {
		t.Errorf("default competitors.count wrong: %d", c.Competitors.Count)
	}
}

func TestLoadYAML(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "c.yaml")
	yaml := `
companies:
  - id: company-x
    address: "x:50051"
auction:
  rate_per_sec: 50
  deadline_ms: 5
  impression_value: 200
competitors:
  count: 2
  base_median: 90
metrics_port: 9300
`
	if err := os.WriteFile(path, []byte(yaml), 0644); err != nil {
		t.Fatal(err)
	}
	c, err := Load(path)
	if err != nil {
		t.Fatalf("load failed: %v", err)
	}
	if len(c.Companies) != 1 || c.Companies[0].ID != "company-x" {
		t.Errorf("companies not parsed: %+v", c.Companies)
	}
	if c.Auction.RatePerSec != 50 || c.Auction.ImpressionValue != 200 {
		t.Errorf("auction not parsed: %+v", c.Auction)
	}
	if c.Competitors.Count != 2 || c.Competitors.BaseMedian != 90 {
		t.Errorf("competitors not parsed: %+v", c.Competitors)
	}
	if c.MetricsPort != 9300 {
		t.Errorf("metrics_port not parsed: %d", c.MetricsPort)
	}
	// a field not in the YAML should keep its default
	if c.Auction.DailyBudget != 200000 {
		t.Errorf("daily_budget default lost: %f", c.Auction.DailyBudget)
	}
	if c.Auction.DayAuctions != 2000 {
		t.Errorf("day_auctions default lost: %d", c.Auction.DayAuctions)
	}
}

func TestPerCompanyBudgetOverride(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "c.yaml")
	yaml := `
companies:
  - id: company-a
    address: "a:50051"
  - id: company-d
    address: "d:50051"
    budget: 6000
auction:
  daily_budget: 12000
`
	if err := os.WriteFile(path, []byte(yaml), 0644); err != nil {
		t.Fatal(err)
	}
	c, err := Load(path)
	if err != nil {
		t.Fatalf("load failed: %v", err)
	}
	shared := c.Auction.DailyBudget
	// A has no override -> falls back to the shared budget
	if got := c.Companies[0].BudgetFor(shared); got != 12000 {
		t.Errorf("company-a should use shared budget 12000, got %f", got)
	}
	// D has an override -> uses its own smaller budget
	if got := c.Companies[1].BudgetFor(shared); got != 6000 {
		t.Errorf("company-d should use its own budget 6000, got %f", got)
	}
}

func TestEnvOverrides(t *testing.T) {
	t.Setenv("RATE_PER_SEC", "0")
	t.Setenv("COMPETITORS_COUNT", "16")
	t.Setenv("IMPRESSION_VALUE", "200")
	c, err := Load("/no/such/file.yaml")
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if c.Auction.RatePerSec != 0 {
		t.Errorf("RATE_PER_SEC override failed: %d", c.Auction.RatePerSec)
	}
	if c.Competitors.Count != 16 {
		t.Errorf("COMPETITORS_COUNT override failed: %d", c.Competitors.Count)
	}
	if c.Auction.ImpressionValue != 200 {
		t.Errorf("IMPRESSION_VALUE override failed: %f", c.Auction.ImpressionValue)
	}
}
