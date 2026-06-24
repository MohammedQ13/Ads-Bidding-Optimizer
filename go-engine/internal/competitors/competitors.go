// Package competitors makes up the "rest of the market" for each auction. These
// are stand-in bidders the C++ companies compete against, drawn from a price
// distribution roughly calibrated to the real iPinYou clearing prices (median
// around 75 fen). Without them, the companies would only ever bid against each
// other.
package competitors

import (
	"math"
	"math/rand"
)

// Pool generates competitor bids for an auction.
type Pool struct {
	count      int
	baseMedian float64
	spread     float64
	rng        *rand.Rand
}

func New(count int, baseMedian, spread float64, seed int64) *Pool {
	return &Pool{
		count:      count,
		baseMedian: baseMedian,
		spread:     spread,
		rng:        rand.New(rand.NewSource(seed)),
	}
}

// Archetype gives each synthetic competitor a personality so the field is not
// all the same. The multiplier scales its draw; the label is for the UI.
type Archetype struct {
	Label string
	Mult  float64
}

// A wider, named set of archetypes so a bigger field still looks varied. The
// pool cycles through these, so count can be larger than the list.
var archetypes = []Archetype{
	{"whale", 1.6},
	{"aggressive", 1.4},
	{"contender", 1.15},
	{"market", 1.0},
	{"value", 0.85},
	{"conservative", 0.7},
	{"bargain", 0.55},
	{"sniper", 1.25},
}

// ArchetypeFor returns the archetype assigned to competitor i.
func ArchetypeFor(i int) Archetype {
	return archetypes[i%len(archetypes)]
}

// Count is how many competitors bid each auction.
func (p *Pool) Count() int { return p.count }

// Bid is one competitor's bid for an auction, tagged with its archetype so the
// engine and UI can show what kind of bidder it was.
type Bid struct {
	Index     int
	Archetype string
	Mult      float64
	Amount    float64
}

// Bids returns this auction's competitor bids (in fen). Each is a draw from a
// lognormal around the base median, scaled by an archetype.
func (p *Pool) Bids() []Bid {
	mu := math.Log(p.baseMedian)
	out := make([]Bid, 0, p.count)
	for i := 0; i < p.count; i++ {
		a := ArchetypeFor(i)
		draw := math.Exp(p.rng.NormFloat64()*p.spread + mu)
		bid := draw * a.Mult
		if bid < 1 {
			bid = 1
		}
		out = append(out, Bid{Index: i, Archetype: a.Label, Mult: a.Mult, Amount: bid})
	}
	return out
}
