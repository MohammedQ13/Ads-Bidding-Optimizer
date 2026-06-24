package competitors

import "testing"

func TestBidsCountAndPositive(t *testing.T) {
	p := New(4, 75, 0.6, 1)
	bids := p.Bids()
	if len(bids) != 4 {
		t.Fatalf("expected 4 bids, got %d", len(bids))
	}
	for i, b := range bids {
		if b.Amount < 1 {
			t.Errorf("bid %d is below the floor of 1: %f", i, b.Amount)
		}
		if b.Archetype == "" {
			t.Errorf("bid %d has no archetype label", i)
		}
	}
}

func TestBidsVary(t *testing.T) {
	// with several competitors the archetypes should give a spread, not all equal
	p := New(5, 75, 0.6, 7)
	bids := p.Bids()
	allSame := true
	for i := 1; i < len(bids); i++ {
		if bids[i].Amount != bids[0].Amount {
			allSame = false
			break
		}
	}
	if allSame {
		t.Errorf("expected competitor bids to vary, got all %f", bids[0].Amount)
	}
}

func TestZeroCompetitors(t *testing.T) {
	p := New(0, 75, 0.6, 1)
	if len(p.Bids()) != 0 {
		t.Errorf("expected no bids when count is 0")
	}
}

func TestDeterministicWithSeed(t *testing.T) {
	a := New(4, 75, 0.6, 42).Bids()
	b := New(4, 75, 0.6, 42).Bids()
	for i := range a {
		if a[i].Amount != b[i].Amount {
			t.Errorf("same seed should give same bids: %f vs %f", a[i].Amount, b[i].Amount)
		}
	}
}

func TestArchetypeForCycles(t *testing.T) {
	// asking past the end of the archetype list should wrap around, not panic
	first := ArchetypeFor(0)
	wrapped := ArchetypeFor(len(archetypes))
	if first.Label != wrapped.Label {
		t.Errorf("archetype %d should match %d (wrap): %s vs %s",
			0, len(archetypes), first.Label, wrapped.Label)
	}
}
