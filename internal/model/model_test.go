package model

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"rinha-fraud/internal/vectorize"
)

// TestParityWithXGBoost is the correctness anchor: the Go Score must reproduce
// xgboost's predict_proba within a tiny tolerance for the sample saved by
// train/train.py. If this passes, the tree walk + sigmoid + base_margin are
// faithful to the trained model. Skips when the artifacts are absent (not yet
// trained on this machine).
func TestParityWithXGBoost(t *testing.T) {
	modelBytes, err := os.ReadFile("../../resources/model.json")
	if err != nil {
		t.Skipf("model.json absent (run train/train.py): %v", err)
	}
	m, err := LoadModel(modelBytes)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	pb, err := os.ReadFile("../../resources/model_parity.json")
	if err != nil {
		t.Skipf("model_parity.json absent (run train/train.py): %v", err)
	}
	var p struct {
		Vectors [][]float64 `json:"vectors"`
		Proba   []float64   `json:"proba"`
	}
	if err := json.Unmarshal(pb, &p); err != nil {
		t.Fatalf("parse parity: %v", err)
	}
	if len(p.Vectors) == 0 || len(p.Vectors) != len(p.Proba) {
		t.Fatalf("bad parity: %d vectors, %d probas", len(p.Vectors), len(p.Proba))
	}

	maxDiff, sum := 0.0, 0.0
	var nOver3, nOver1 int
	worst := -1
	for i, v := range p.Vectors {
		if len(v) != vectorize.Dims {
			t.Fatalf("sample %d has %d dims", i, len(v))
		}
		var q [vectorize.Dims]float64
		copy(q[:], v)
		got := m.Score(q)
		d := math.Abs(got - p.Proba[i])
		sum += d
		if d > 1e-3 {
			nOver3++
		}
		if d > 0.1 {
			nOver1++
		}
		if d > maxDiff {
			maxDiff, worst = d, i
		}
	}
	t.Logf("parity over %d: mean=%.3e max=%.3e  (>1e-3: %d, >0.1: %d)",
		len(p.Vectors), sum/float64(len(p.Vectors)), maxDiff, nOver3, nOver1)
	if worst >= 0 {
		var q [vectorize.Dims]float64
		copy(q[:], p.Vectors[worst])
		t.Logf("worst sample %d: Go=%.6f xgboost=%.6f", worst, m.Score(q), p.Proba[worst])
	}
	if maxDiff > 1e-5 {
		t.Fatalf("parity max diff %.3e > 1e-5 — Go inference diverges from xgboost", maxDiff)
	}
}

// TestLoadRejectsGarbage: a non-model byte blob must error, not panic.
func TestLoadRejectsGarbage(t *testing.T) {
	if _, err := LoadModel([]byte("not a model")); err == nil {
		t.Fatal("expected error loading garbage")
	}
}
