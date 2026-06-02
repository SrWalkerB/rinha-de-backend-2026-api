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
	got, _, _ := ix.searchTopK(q)
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

// Adaptive two-tier: with a tiny cheap nprobe but escalation forced on (trigger
// radius ~0 => every query re-runs at a high nprobe), the escalated path must
// still reproduce the exact brute 5-NN, and escalation must actually fire.
func TestAdaptiveEscalatedExact(t *testing.T) {
	rng := rand.New(rand.NewSource(21))
	const N, M, nlist = 12000, 1000, 16
	rows := make([]row, N)
	for i := range rows {
		rows[i] = row{randVec(rng), rng.Float64() < 0.44}
	}
	b := NewBuilder(N, nlist, 5) // nlist>1 so a cheap nprobe leaves cells unscanned
	for _, r := range rows {
		b.Add(r.v, r.fraud)
	}
	ix := b.Build()
	ix.SetNProbe(1)           // cheap pass scans only 1 of nlist cells/bucket
	ix.SetTriggerRadius(1e-9) // ...but every query escalates
	ix.SetTriggerMargin(-1)   // radius alone decides
	ix.SetNProbeHigh(nlist)   // high pass = full within-bucket => exact
	fired := 0
	for j := 0; j < M; j++ {
		q := randVec(rng)
		got, _, esc := ix.searchTopK(&q)
		if esc {
			fired++
		}
		want := bruteTopK(ix, &q)
		if got.dist != want.dist || got.fraud != want.fraud {
			t.Fatalf("escalated 5-NN != brute\n got dist=%v fraud=%v\nwant dist=%v fraud=%v",
				got.dist, got.fraud, want.dist, want.fraud)
		}
	}
	if fired == 0 {
		t.Fatal("escalation never fired; adaptive path was not exercised")
	}
}

func TestScoreScanTraceReportsCheapVote(t *testing.T) {
	rng := rand.New(rand.NewSource(22))
	const N, nlist = 12000, 16
	rows := make([]row, N)
	for i := range rows {
		rows[i] = row{randVec(rng), rng.Float64() < 0.44}
	}
	b := NewBuilder(N, nlist, 5)
	for _, r := range rows {
		b.Add(r.v, r.fraud)
	}
	ix := b.Build()
	ix.SetNProbe(1)
	ix.SetTriggerRadius(1e-9)
	ix.SetTriggerMargin(-1)
	ix.SetNProbeHigh(nlist)

	q := randVec(rng)
	score, scanned, tr := ix.ScoreScanTrace(q)
	if !tr.Escalated {
		t.Fatal("trace did not report forced escalation")
	}
	if tr.CheapFraudCount < 0 || tr.CheapFraudCount > K {
		t.Fatalf("cheap fraud count = %d, want 0..%d", tr.CheapFraudCount, K)
	}
	if scanned <= 0 || tr.CheapScanned <= 0 || tr.HighScanned <= 0 {
		t.Fatalf("scan counts not populated: scanned=%d trace=%+v", scanned, tr)
	}
	if score < 0 || score > 1 {
		t.Fatalf("score = %v, want [0,1]", score)
	}
}

func TestNProbeHighAllowsWideEscalation(t *testing.T) {
	b := NewBuilder(10, 512, 1)
	for i := 0; i < 10; i++ {
		b.Add(randVec(rand.New(rand.NewSource(int64(i)))), false)
	}
	ix := b.Build()
	ix.SetNProbe(16)
	ix.SetNProbeHigh(512)
	if ix.nprobeHigh != 512 {
		t.Fatalf("nprobeHigh = %d, want 512", ix.nprobeHigh)
	}
}

func TestNProbeHighForCountOverridesDefault(t *testing.T) {
	b := NewBuilder(10, 128, 1)
	for i := 0; i < 10; i++ {
		b.Add(randVec(rand.New(rand.NewSource(int64(i)))), false)
	}
	ix := b.Build()
	ix.SetNProbe(16)
	ix.SetNProbeHigh(96)
	ix.SetNProbeHighForCount(2, 128)
	ix.SetNProbeHighForCount(4, 32)

	if got := ix.highProbeForCount(2); got != 128 {
		t.Fatalf("highProbeForCount(2) = %d, want 128", got)
	}
	if got := ix.highProbeForCount(3); got != 96 {
		t.Fatalf("highProbeForCount(3) = %d, want 96", got)
	}
	if got := ix.highProbeForCount(4); got != 32 {
		t.Fatalf("highProbeForCount(4) = %d, want 32", got)
	}
}
