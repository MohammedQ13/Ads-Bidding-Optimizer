package stats

import "testing"

// helper: record one auction where company id wins at bid (profit V-bid) and the
// listed losers each bid but lose.
func recordWin(s *Store, winner string, bid, profit float64, budget float64, losers ...string) {
	entries := []BidEntry{
		{ID: winner, OK: true, Won: true, Bid: bid, Profit: profit, Spend: bid, Budget: budget},
	}
	for _, l := range losers {
		entries = append(entries, BidEntry{ID: l, OK: true, Won: false, Bid: bid - 5})
	}
	s.Record(Auction{RequestID: "x", Clearing: bid, Winner: winner, V: 150, Entries: entries})
}

func TestRecordWinLoss(t *testing.T) {
	s := New()
	s.SetBudget("a", 1000)
	// two wins, then an auction where "a" loses, then one where it does not bid
	recordWin(s, "a", 100, 50, 900)
	recordWin(s, "a", 120, 30, 780)
	s.Record(Auction{RequestID: "x", Clearing: 100, Winner: "market", V: 150,
		Entries: []BidEntry{{ID: "a", OK: true, Won: false, Bid: 40}}})
	s.Record(Auction{RequestID: "x", Clearing: 110, Winner: "market", V: 150,
		Entries: []BidEntry{{ID: "a", OK: false}}})

	snap := s.Snapshot()
	if snap.Auctions != 4 {
		t.Errorf("auctions: got %d want 4", snap.Auctions)
	}
	if snap.LastClearing != 110 {
		t.Errorf("last clearing: got %f want 110", snap.LastClearing)
	}
	c := snap.Companies["a"]
	if c.Won != 2 || c.Lost != 1 || c.NoBid != 1 {
		t.Errorf("counts wrong: %+v", c)
	}
	if c.Profit != 80 || c.Spend != 220 || c.Budget != 780 {
		t.Errorf("totals wrong: %+v", c)
	}
}

func TestSnapshotIsCopy(t *testing.T) {
	s := New()
	recordWin(s, "a", 10, 10, 0)
	snap := s.Snapshot()
	// mutate the store after taking the snapshot
	recordWin(s, "a", 10, 10, 0)
	if snap.Companies["a"].Won != 1 {
		t.Errorf("snapshot should not change when the store does: %d", snap.Companies["a"].Won)
	}
}

func TestEventFeedAndHistograms(t *testing.T) {
	s := New()
	s.Record(Auction{
		RequestID:      "auc-1",
		Hour:           14,
		Exchange:       2,
		Slot:           "300x250",
		Clearing:       75,
		CompetitorTop:  75,
		CompetitorBids: []CompetitorBid{{Archetype: "aggressive", Amount: 75}},
		Winner:         "company-b",
		V:              150,
		Entries: []BidEntry{
			{ID: "company-b", OK: true, Won: false, Bid: 60, WinProb: 0.4, ExpProfit: 36, InfUs: 120, RttUs: 800},
		},
	})
	snap := s.Snapshot()
	if len(snap.Events) != 1 {
		t.Fatalf("expected 1 event, got %d", len(snap.Events))
	}
	if snap.Events[0].Winner != "company-b" {
		t.Errorf("event winner wrong: %+v", snap.Events[0])
	}
	if len(snap.Archetypes) != 1 || snap.Archetypes[0].Label != "aggressive" {
		t.Errorf("archetype not tallied: %+v", snap.Archetypes)
	}
	if snap.Impressions.ByHour[14] != 1 {
		t.Errorf("hour bucket not counted: %+v", snap.Impressions.ByHour)
	}
	c := snap.Companies["company-b"]
	if c.Bids != 1 || c.LastBid != 60 {
		t.Errorf("bid stats wrong: %+v", c)
	}
	if c.Inference.P50 == 0 {
		t.Errorf("inference latency should be recorded: %+v", c.Inference)
	}
}
