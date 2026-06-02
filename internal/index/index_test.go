package index

import (
	"math"
	"testing"
)

// --- helpers --------------------------------------------------------------

type row struct {
	v     [Dims]float64
	fraud bool
}

func mkvec(base float64) [Dims]float64 {
	var v [Dims]float64
	for i := range v {
		v[i] = base
	}
	return v
}

// buildIndex builds an EXACT index for tests: nlist=1 means one cell per bucket,
// so a query scans its whole bucket (plus any bucket the cross-bucket bound
// admits) — identical 5-NN to a full brute scan.
func buildIndex(rows []row) *Index {
	b := NewBuilder(len(rows), 1, 1)
	for _, r := range rows {
		b.Add(r.v, r.fraud)
	}
	return b.Build()
}

func repeat(v [Dims]float64, fraud bool, n int) []row {
	rows := make([]row, n)
	for i := range rows {
		rows[i] = row{v, fraud}
	}
	return rows
}

// bruteScore is the exact O(n) reference: scan every stored row. Used as the
// oracle the partitioned search must match.
func bruteScore(ix *Index, q *[Dims]float64) float64 {
	tk := newTopK()
	for r := 0; r < ix.n; r++ {
		tk.consider(ix.dist2(q, r), ix.isFraud(r))
	}
	return tk.fraudScore()
}

// --- quantization ---------------------------------------------------------

func TestQuantizeMapping(t *testing.T) {
	if got := quantize(-1); got != 0 {
		t.Errorf("quantize(-1) = %d, want 0", got)
	}
	if got := quantize(0); got != 1 {
		t.Errorf("quantize(0) = %d, want 1", got)
	}
	if got := quantize(1); got != 10001 {
		t.Errorf("quantize(1) = %d, want 10001", got)
	}
	if quantize(0.5) <= quantize(0.25) {
		t.Errorf("quantize not monotonic: q(0.5)=%d q(0.25)=%d", quantize(0.5), quantize(0.25))
	}
}

func TestDequantRoundTrip(t *testing.T) {
	if got := dequant(0); got != -1 {
		t.Errorf("dequant(0) = %v, want -1 (sentinel)", got)
	}
	for _, v := range []float64{0, 0.0001, 0.0833, 0.3913, 0.8261, 0.9999, 1} {
		if got := dequant(quantize(v)); got != v {
			t.Errorf("dequant(quantize(%v)) = %v, want exact", v, got)
		}
	}
}

// --- partition keys -------------------------------------------------------

func TestBucketOf(t *testing.T) {
	var v [Dims]float64 // all zero, non-null
	if b := bucketOf(&v); b != 0 {
		t.Errorf("all-zero bucket = %d, want 0", b)
	}
	v[9], v[10], v[11] = 1, 1, 1
	v[5], v[6] = -1, -1 // null
	if b := bucketOf(&v); b != 15 {
		t.Errorf("all-set+null bucket = %d, want 15", b)
	}
}

// hour 23 and hour 0 are FAR apart (linear encoding), never treated adjacent —
// the search relies on the full euclidean distance over these dims.
func TestHourEncodingLinear(t *testing.T) {
	var v0, v23 [Dims]float64
	v0[3] = 0.0 / 23
	v23[3] = 23.0 / 23
	d := (v0[3] - v23[3]) * (v0[3] - v23[3])
	if math.Abs(d-1.0) > 1e-9 {
		t.Errorf("hour 0 vs 23 squared dist = %v, want 1.0", d)
	}
}

// --- scoring (exact 5-NN decisions) --------------------------------------

func TestScoreAllFraud(t *testing.T) {
	ix := buildIndex(repeat(mkvec(0.5), true, 5))
	if s := ix.Score(mkvec(0.5)); s != 1.0 {
		t.Errorf("Score = %v, want 1.0", s)
	}
}

func TestScoreAllLegit(t *testing.T) {
	ix := buildIndex(repeat(mkvec(0.5), false, 5))
	if s := ix.Score(mkvec(0.5)); s != 0.0 {
		t.Errorf("Score = %v, want 0.0", s)
	}
}

// 3 fraud + 2 legit among 5 nearest -> 0.6 -> not approved (0.6 < 0.6 == false).
func TestScoreMixedAtThreshold(t *testing.T) {
	rows := append(repeat(mkvec(0.5), true, 3), repeat(mkvec(0.5), false, 2)...)
	ix := buildIndex(rows)
	s := ix.Score(mkvec(0.5))
	if math.Abs(s-0.6) > 1e-9 {
		t.Fatalf("Score = %v, want 0.6", s)
	}
	if s < Threshold {
		t.Errorf("score 0.6 must NOT be approved (Threshold=%v)", Threshold)
	}
}

func TestScoreNearestSelection(t *testing.T) {
	rows := append(repeat(mkvec(0.10), true, 5), repeat(mkvec(0.90), false, 10)...)
	ix := buildIndex(rows)
	if s := ix.Score(mkvec(0.10)); s != 1.0 {
		t.Errorf("query near fraud cluster: Score = %v, want 1.0", s)
	}
	if s := ix.Score(mkvec(0.90)); s != 0.0 {
		t.Errorf("query near legit cluster: Score = %v, want 0.0", s)
	}
}

func TestSentinelMatchesSentinel(t *testing.T) {
	fraudVec := mkvec(0.9)
	fraudVec[5], fraudVec[6] = -1, -1 // no last_transaction
	legitVec := mkvec(0.9)
	legitVec[5], legitVec[6] = 0.5, 0.5 // present last_transaction
	rows := append(repeat(fraudVec, true, 5), repeat(legitVec, false, 5)...)
	ix := buildIndex(rows)
	if s := ix.Score(fraudVec); s != 1.0 {
		t.Errorf("sentinel query: Score = %v, want 1.0", s)
	}
}

func TestScoreEmptyIndex(t *testing.T) {
	ix := buildIndex(nil)
	if s := ix.Score(mkvec(0.5)); s != 0 {
		t.Errorf("empty index Score = %v, want 0", s)
	}
}
