package knn

import (
	"math/rand"
	"testing"

	"rinha-fraud/internal/vectorize"
)

// buildBenchIndex makes a deterministic pseudo-random index of n vectors so the
// benchmark is reproducible across runs (same seed => same data => comparable ns/op).
func buildBenchIndex(n int) *Index {
	r := rand.New(rand.NewSource(42))
	ix := NewIndex(n)
	for i := 0; i < n; i++ {
		var v [vectorize.Dims]float64
		for d := range v {
			v[d] = r.Float64()
		}
		ix.Add(v, r.Intn(2) == 0)
	}
	return ix
}

func benchQuery() [vectorize.Dims]float64 {
	r := rand.New(rand.NewSource(7))
	var q [vectorize.Dims]float64
	for d := range q {
		q[d] = r.Float64()
	}
	return q
}

const benchN = 200_000

// Run all three:
//   go test -bench=Score -benchmem -run=^$ ./internal/knn

func BenchmarkScoreBrute(b *testing.B) {
	ix := buildBenchIndex(benchN) // un-Built => brute force
	q := benchQuery()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ix.Score(q)
	}
}

func BenchmarkScoreVPTree(b *testing.B) {
	ix := buildBenchIndex(benchN)
	ix.Build(BuildConfig{Mode: "vptree"})
	q := benchQuery()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ix.Score(q)
	}
}

func BenchmarkScoreIVF(b *testing.B) {
	ix := buildBenchIndex(benchN)
	ix.Build(BuildConfig{Mode: "ivf", NList: 256, NProbe: 8, Iters: 8})
	q := benchQuery()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ix.Score(q)
	}
}
