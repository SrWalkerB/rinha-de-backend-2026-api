package index

import (
	"math/rand"
	"testing"
)

// randVec produces a realistically-shaped reference/query vector: discrete dims
// on their grids, continuous dims with full float64 entropy (so exact-distance
// ties at the 5-NN boundary do not occur).
func randVec(rng *rand.Rand) [Dims]float64 {
	var v [Dims]float64
	v[0] = rng.Float64()
	v[1] = float64(rng.Intn(13)) / 12
	v[2] = rng.Float64()
	v[3] = float64(rng.Intn(24)) / 23
	v[4] = float64(rng.Intn(7)) / 6
	if rng.Float64() < 0.3 {
		v[5], v[6] = -1, -1 // null last_transaction
	} else {
		v[5] = rng.Float64()
		v[6] = rng.Float64()
	}
	v[7] = rng.Float64()
	v[8] = float64(rng.Intn(21)) / 20
	v[9] = float64(rng.Intn(2))
	v[10] = float64(rng.Intn(2))
	v[11] = float64(rng.Intn(2))
	mccs := []float64{0.15, 0.25, 0.30, 0.45, 0.5, 0.75, 0.85}
	v[12] = mccs[rng.Intn(len(mccs))]
	v[13] = rng.Float64()
	return v
}

func bruteTopK(ix *Index, q *[Dims]float64) topK {
	tk := newTopK()
	for r := 0; r < ix.n; r++ {
		tk.consider(ix.dist2(q, r), ix.isFraud(r))
	}
	return tk
}

// assertExact fails if the partitioned search does not return the identical
// top-5 (distances AND fraud labels) as the full brute scan.
func assertExact(t *testing.T, ix *Index, q *[Dims]float64, label string) {
	t.Helper()
	got, _ := ix.searchTopK(q)
	want := bruteTopK(ix, q)
	if got.dist != want.dist || got.fraud != want.fraud {
		t.Fatalf("%s: partitioned 5-NN != brute\n got dist=%v fraud=%v\nwant dist=%v fraud=%v",
			label, got.dist, got.fraud, want.dist, want.fraud)
	}
}

// The core guarantee: over many random references and queries the partitioned
// search reproduces the exact float64 brute 5-NN — same neighbors, same score.
func TestExactVsBrute(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	const N, M = 12000, 4000

	rows := make([]row, N)
	for i := range rows {
		rows[i] = row{randVec(rng), rng.Float64() < 0.44}
	}
	ix := buildIndex(rows)

	for i := 0; i < M; i++ {
		q := randVec(rng)
		assertExact(t, ix, &q, "random query")
	}
}

// Sparse buckets force cross-bucket expansion: with very few references, the
// query's own bucket rarely holds 5 neighbors, so the bound-checked fallback to
// other buckets must still reproduce the exact 5-NN.
func TestExactSparseCrossBucket(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for trial := 0; trial < 200; trial++ {
		n := 5 + rng.Intn(25) // 5..29 refs scattered across 16 buckets
		rows := make([]row, n)
		for i := range rows {
			rows[i] = row{randVec(rng), rng.Float64() < 0.5}
		}
		ix := buildIndex(rows)
		for j := 0; j < 20; j++ {
			q := randVec(rng)
			assertExact(t, ix, &q, "sparse query")
		}
	}
}

// Explicit boundary queries: all-null, all-non-null, hour/weekday extremes, and
// every hard-bucket signature.
func TestExactBoundaries(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	rows := make([]row, 6000)
	for i := range rows {
		rows[i] = row{randVec(rng), rng.Float64() < 0.44}
	}
	ix := buildIndex(rows)

	var queries [][Dims]float64
	// every bucket signature, at hour/weekday extremes
	for b := 0; b < numBuckets; b++ {
		for _, hw := range [][2]float64{{0, 0}, {23.0 / 23, 6.0 / 6}, {12.0 / 23, 3.0 / 6}} {
			q := randVec(rng)
			q[9] = float64((b >> 3) & 1)
			q[10] = float64((b >> 2) & 1)
			q[11] = float64((b >> 1) & 1)
			if b&1 == 1 {
				q[5], q[6] = -1, -1
			} else {
				q[5], q[6] = rng.Float64(), rng.Float64()
			}
			q[3], q[4] = hw[0], hw[1]
			queries = append(queries, q)
		}
	}
	for i := range queries {
		assertExact(t, ix, &queries[i], "boundary query")
	}
}

// maxScan is a guardrail only: with it unset (0) results are exact; the test
// confirms the unlimited default reproduces brute even on sparse data.
func TestMaxScanUnlimitedIsExact(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	rows := make([]row, 200)
	for i := range rows {
		rows[i] = row{randVec(rng), rng.Float64() < 0.5}
	}
	ix := buildIndex(rows)
	if ix.maxScan != 0 {
		t.Fatalf("default maxScan = %d, want 0 (unlimited)", ix.maxScan)
	}
	for j := 0; j < 300; j++ {
		q := randVec(rng)
		assertExact(t, ix, &q, "unlimited query")
	}
}
