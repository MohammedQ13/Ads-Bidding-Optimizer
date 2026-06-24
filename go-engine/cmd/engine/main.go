// The auction engine entry point. Loads config, starts the metrics endpoint,
// waits a moment for the company servers to come up, then runs auctions until it
// gets a shutdown signal.
package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"rtbengine/internal/config"
	"rtbengine/internal/engine"
	"rtbengine/internal/stats"
)

func main() {
	// optional: --replay <file> runs a backtest over a recorded auction log
	// instead of live auctions. Usage: engine [config.yaml] [--replay file]
	cfgPath := "config.yaml"
	replayFile := ""
	args := os.Args[1:]
	for i := 0; i < len(args); i++ {
		if args[i] == "--replay" && i+1 < len(args) {
			replayFile = args[i+1]
			i++
		} else if cfgPath == "config.yaml" {
			cfgPath = args[i]
		}
	}
	cfg, err := config.Load(cfgPath)
	if err != nil {
		log.Fatalf("config: %v", err)
	}
	if len(cfg.Companies) == 0 {
		log.Fatalf("no companies configured in %s", cfgPath)
	}

	log.Printf("auction engine starting: %d companies, %d competitors/auction, V=%.0f",
		len(cfg.Companies), cfg.Competitors.Count, cfg.Auction.ImpressionValue)

	st := stats.New()
	st.Configure(cfg.Auction.ImpressionValue, cfg.Auction.DeadlineMs, cfg.Competitors.Count)
	st.Serve(cfg.MetricsPort)
	log.Printf("stats API on :%d (/stats JSON, /events SSE)", cfg.MetricsPort)

	// fixed seed keeps runs reproducible; vary it via config if needed later
	eng, err := engine.New(cfg, st, 12345)
	if err != nil {
		log.Fatalf("engine: %v", err)
	}
	defer eng.Close()

	// give the company servers a few seconds to finish starting up
	time.Sleep(3 * time.Second)

	ctx, cancel := context.WithCancel(context.Background())
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		log.Printf("shutting down")
		cancel()
	}()

	// background telemetry collectors: scrape each C++ /metrics page and watch
	// the retrainer status file, both folded into the live /stats feed.
	eng.StartCollectors(ctx)

	if replayFile != "" {
		log.Printf("replay/backtest mode: %s", replayFile)
		if err := eng.RunReplay(ctx, replayFile); err != nil {
			log.Printf("replay error: %v", err)
		}
		snap := st.Snapshot()
		log.Printf("replay done: %d auctions", snap.Auctions)
		for id, c := range snap.Companies {
			total := c.Won + c.Lost
			wr := 0.0
			if total > 0 {
				wr = float64(c.Won) / float64(total)
			}
			log.Printf("  %s: win_rate %.3f  profit %.0f  (won %d / %d)",
				id, wr, c.Profit, c.Won, total)
		}
	} else {
		eng.Run(ctx)
	}
	log.Printf("stopped")
}
