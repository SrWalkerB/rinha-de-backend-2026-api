package knn

import (
	"math"
	"testing"

	"rinha-fraud/internal/vectorize"
)

func mkvec(base float64) [vectorize.Dims]float64 {
	var v [vectorize.Dims]float64
	for i := range v {
		v[i] = base
	}
	return v
}

// quantize pins the uint16 mapping: -1 sentinel -> 0, [0,1] -> [1,65535], monotonic.
func TestQuantizeMapping(t *testing.T) {
	if got := quantize(-1); got != 0 {
		t.Errorf("quantize(-1) = %d, want 0", got)
	}
	if got := quantize(0); got != 1 {
		t.Errorf("quantize(0) = %d, want 1", got)
	}
	if got := quantize(1); got != 65535 {
		t.Errorf("quantize(1) = %d, want 65535", got)
	}
	if quantize(0.5) <= quantize(0.25) {
		t.Errorf("quantize not monotonic: q(0.5)=%d q(0.25)=%d", quantize(0.5), quantize(0.25))
	}
}

func TestScoreAllFraud(t *testing.T) {
	ix := NewIndex(5)
	for i := 0; i < 5; i++ {
		ix.Add(mkvec(0.5), true)
	}
	if s := ix.Score(mkvec(0.5)); s != 1.0 {
		t.Errorf("Score = %v, want 1.0", s)
	}
}

func TestScoreAllLegit(t *testing.T) {
	ix := NewIndex(5)
	for i := 0; i < 5; i++ {
		ix.Add(mkvec(0.5), false)
	}
	if s := ix.Score(mkvec(0.5)); s != 0.0 {
		t.Errorf("Score = %v, want 0.0", s)
	}
}

// 3 fraud + 2 legit among 5 nearest -> 0.6 -> approved is false (0.6 < 0.6 == false).
func TestScoreMixedAtThreshold(t *testing.T) {
	ix := NewIndex(5)
	ix.Add(mkvec(0.5), true)
	ix.Add(mkvec(0.5), true)
	ix.Add(mkvec(0.5), true)
	ix.Add(mkvec(0.5), false)
	ix.Add(mkvec(0.5), false)
	s := ix.Score(mkvec(0.5))
	if math.Abs(s-0.6) > 1e-9 {
		t.Fatalf("Score = %v, want 0.6", s)
	}
	if s < Threshold {
		t.Errorf("score 0.6 must NOT be approved (Threshold=%v)", Threshold)
	}
}

// The 5 nearest are selected out of a larger set.
func TestScoreNearestSelection(t *testing.T) {
	ix := NewIndex(20)
	for i := 0; i < 5; i++ { // 5 fraud clustered near 0.10
		ix.Add(mkvec(0.10), true)
	}
	for i := 0; i < 10; i++ { // 10 legit far away near 0.90
		ix.Add(mkvec(0.90), false)
	}
	if s := ix.Score(mkvec(0.10)); s != 1.0 {
		t.Errorf("query near fraud cluster: Score = %v, want 1.0", s)
	}
	if s := ix.Score(mkvec(0.90)); s != 0.0 {
		t.Errorf("query near legit cluster: Score = %v, want 0.0", s)
	}
}

func TestSentinelMatchesSentinel(t *testing.T) {
	ix := NewIndex(10)
	// fraud vectors that share the -1 sentinel in dims 5,6 (no last_transaction)
	fraudVec := mkvec(0.9)
	fraudVec[5], fraudVec[6] = -1, -1
	for i := 0; i < 5; i++ {
		ix.Add(fraudVec, true)
	}
	// legit vectors with present last_transaction (dims 5,6 in [0,1])
	legitVec := mkvec(0.9)
	legitVec[5], legitVec[6] = 0.5, 0.5
	for i := 0; i < 5; i++ {
		ix.Add(legitVec, false)
	}
	// query also has the sentinel -> should land near the fraud (sentinel) cluster
	if s := ix.Score(fraudVec); s != 1.0 {
		t.Errorf("sentinel query: Score = %v, want 1.0", s)
	}
}
