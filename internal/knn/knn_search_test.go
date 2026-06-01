package knn

import (
	"math/rand"
	"testing"

	"rinha-fraud/internal/vectorize"
)

// fillClustered adds n points grouped into `clusters` tight blobs, so the IVF
// k-means has real structure to recover (mirrors how the real dataset clusters).
func fillClustered(ix *Index, n, clusters int, seed int64) {
	r := rand.New(rand.NewSource(seed))
	centers := make([][vectorize.Dims]float64, clusters)
	for c := range centers {
		for d := range centers[c] {
			centers[c][d] = r.Float64()
		}
	}
	for i := 0; i < n; i++ {
		c := r.Intn(clusters)
		var v [vectorize.Dims]float64
		for d := range v {
			x := centers[c][d] + (r.Float64()-0.5)*0.08
			if x < 0 {
				x = 0
			} else if x > 1 {
				x = 1
			}
			v[d] = x
		}
		ix.Add(v, c%2 == 0) // fraud label by cluster parity
	}
}

func randomQueries(n int, seed int64) [][vectorize.Dims]float64 {
	r := rand.New(rand.NewSource(seed))
	qs := make([][vectorize.Dims]float64, n)
	for i := range qs {
		for d := range qs[i] {
			qs[i][d] = r.Float64()
		}
	}
	return qs
}

// TestVPTreeExactMatchesBrute: the VP-tree must return the SAME K nearest
// distances as brute force for every query (it is an exact method). Comparing
// the sorted distance multiset is tie-safe.
func TestVPTreeExactMatchesBrute(t *testing.T) {
	const n = 5000
	ref := NewIndex(n)
	fillClustered(ref, n, 40, 1) // un-Built -> brute force

	vp := NewIndex(n)
	fillClustered(vp, n, 40, 1) // same seed -> identical data
	vp.Build(BuildConfig{Mode: "vptree"})
	if vp.mode != modeVPTree {
		t.Fatalf("expected vptree mode, got %d", vp.mode)
	}

	for i, q := range randomQueries(300, 99) {
		q := q // addressable per-iteration copy
		want := ref.bruteTopK(&q)
		got := vp.vp.searchTopK(vp, &q)
		if got.dist != want.dist {
			t.Fatalf("query %d: VP distances %v != brute %v", i, got.dist, want.dist)
		}
	}
}

// TestIVFRecallHigh: IVF is approximate, but on clustered data it should find
// most of the true K nearest. Assert mean recall@K is high.
func TestIVFRecallHigh(t *testing.T) {
	const n = 20000
	ref := NewIndex(n)
	fillClustered(ref, n, 64, 7) // brute reference

	ivf := NewIndex(n)
	fillClustered(ivf, n, 64, 7)
	ivf.Build(BuildConfig{Mode: "ivf", NList: 64, NProbe: 8, Iters: 10})
	if ivf.mode != modeIVF {
		t.Fatalf("expected ivf mode, got %d", ivf.mode)
	}

	queries := randomQueries(300, 123)
	var totalRecall float64
	for _, q := range queries {
		q := q
		want := ref.bruteTopK(&q)
		got := ivf.ivf.searchTopK(ivf, &q)
		totalRecall += recallAtK(want, got)
	}
	mean := totalRecall / float64(len(queries))
	if mean < 0.90 {
		t.Fatalf("IVF mean recall@%d = %.3f, want >= 0.90", K, mean)
	}
	t.Logf("IVF mean recall@%d = %.3f", K, mean)
}

// recallAtK = fraction of the true K distances that also appear in the
// approximate result (multiset overlap).
func recallAtK(want, got topK) float64 {
	hits := 0
	used := [K]bool{}
	for i := 0; i < K; i++ {
		for j := 0; j < K; j++ {
			if !used[j] && got.dist[j] == want.dist[i] {
				used[j] = true
				hits++
				break
			}
		}
	}
	return float64(hits) / float64(K)
}

// TestSmallNStaysBruteEvenWithMode: below buildThreshold, Build must keep brute
// force (so the exact-on-small-data unit tests never hit VP/IVF).
func TestSmallNStaysBruteEvenWithMode(t *testing.T) {
	ix := NewIndex(10)
	fillClustered(ix, 10, 3, 5)
	ix.Build(BuildConfig{Mode: "ivf", NList: 64, NProbe: 8})
	if ix.mode != modeBrute {
		t.Fatalf("small N must stay brute, got mode %d", ix.mode)
	}
}
