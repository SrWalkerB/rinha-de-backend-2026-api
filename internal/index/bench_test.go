package index

import (
	"math/rand"
	"testing"
)

// benchIndex builds a large synthetic index with realistic bucket skew (online
// card-present transactions dominate), so the per-query own-cell scan size is
// representative of production.
func benchIndex(tb testing.TB, n int) *Index {
	tb.Helper()
	rng := rand.New(rand.NewSource(2024))
	b := NewBuilder(n, 1024, 10)
	for i := 0; i < n; i++ {
		v := randVec(rng)
		// Skew: ~80% online, ~75% card-present, ~70% known merchant.
		v[9] = boolf(rng.Float64() < 0.80)
		v[10] = boolf(rng.Float64() < 0.75)
		v[11] = boolf(rng.Float64() < 0.30)
		b.Add(v, rng.Float64() < 0.44)
	}
	ix := b.Build()
	ix.SetNProbe(8)
	return ix
}

func boolf(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

func BenchmarkScore(b *testing.B) {
	ix := benchIndex(b, 1_000_000)
	rng := rand.New(rand.NewSource(55))
	queries := make([][Dims]float64, 1024)
	for i := range queries {
		q := randVec(rng)
		q[9], q[10], q[11] = boolf(rng.Float64() < 0.80), boolf(rng.Float64() < 0.75), boolf(rng.Float64() < 0.30)
		queries[i] = q
	}

	b.ReportAllocs()
	b.ResetTimer()
	var sink float64
	for i := 0; i < b.N; i++ {
		sink += ix.Score(queries[i&1023])
	}
	_ = sink
}

// Score must not allocate on the hot path.
func TestScoreZeroAlloc(t *testing.T) {
	ix := benchIndex(t, 50_000)
	rng := rand.New(rand.NewSource(77))
	q := randVec(rng)
	if avg := testing.AllocsPerRun(2000, func() { _ = ix.Score(q) }); avg != 0 {
		t.Errorf("Score allocs/op = %v, want 0", avg)
	}
}
