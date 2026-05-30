package vectorize

import (
	"encoding/json"
	"math"
	"testing"
)

// stdVectorizer builds a Vectorizer with the official normalization constants
// and mcc_risk table from the challenge docs (DATASET.md).
func stdVectorizer() *Vectorizer {
	norm := Norm{
		MaxAmount:            10000,
		MaxInstallments:      12,
		AmountVsAvgRatio:     10,
		MaxMinutes:           1440,
		MaxKm:                1000,
		MaxTxCount24h:        20,
		MaxMerchantAvgAmount: 10000,
	}
	mcc := map[string]float64{
		"5411": 0.15, "5812": 0.30, "5912": 0.20, "5944": 0.45, "7801": 0.80,
		"7802": 0.75, "7995": 0.85, "4511": 0.35, "5311": 0.25, "5999": 0.50,
	}
	return New(norm, mcc)
}

func assertVector(t *testing.T, got [Dims]float64, want [Dims]float64) {
	t.Helper()
	const tol = 1e-3
	for i := 0; i < Dims; i++ {
		if math.Abs(got[i]-want[i]) > tol {
			t.Errorf("dim %d: got %.4f, want %.4f", i, got[i], want[i])
		}
	}
}

// Legit example from REGRAS_DE_DETECCAO.md (tx-1329056812).
func TestVectorizeLegitExample(t *testing.T) {
	raw := `{
		"id": "tx-1329056812",
		"transaction": { "amount": 41.12, "installments": 2, "requested_at": "2026-03-11T18:45:53Z" },
		"customer": { "avg_amount": 82.24, "tx_count_24h": 3, "known_merchants": ["MERC-003", "MERC-016"] },
		"merchant": { "id": "MERC-016", "mcc": "5411", "avg_amount": 60.25 },
		"terminal": { "is_online": false, "card_present": true, "km_from_home": 29.23 },
		"last_transaction": null
	}`
	var p Payload
	if err := json.Unmarshal([]byte(raw), &p); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	want := [Dims]float64{0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006}
	assertVector(t, stdVectorizer().Vectorize(&p), want)
}

// Fraud example from REGRAS_DE_DETECCAO.md (tx-3330991687).
func TestVectorizeFraudExample(t *testing.T) {
	raw := `{
		"id": "tx-3330991687",
		"transaction": { "amount": 9505.97, "installments": 10, "requested_at": "2026-03-14T05:15:12Z" },
		"customer": { "avg_amount": 81.28, "tx_count_24h": 20, "known_merchants": ["MERC-008", "MERC-007", "MERC-005"] },
		"merchant": { "id": "MERC-068", "mcc": "7802", "avg_amount": 54.86 },
		"terminal": { "is_online": false, "card_present": true, "km_from_home": 952.27 },
		"last_transaction": null
	}`
	var p Payload
	if err := json.Unmarshal([]byte(raw), &p); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	want := [Dims]float64{0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055}
	assertVector(t, stdVectorizer().Vectorize(&p), want)
}

// last_transaction present exercises dims 5 (minutes) and 6 (km_from_last).
func TestVectorizeWithLastTransaction(t *testing.T) {
	raw := `{
		"id": "tx-x",
		"transaction": { "amount": 100, "installments": 1, "requested_at": "2026-03-11T12:00:00Z" },
		"customer": { "avg_amount": 100, "tx_count_24h": 1, "known_merchants": ["MERC-001"] },
		"merchant": { "id": "MERC-001", "mcc": "9999", "avg_amount": 100 },
		"terminal": { "is_online": true, "card_present": false, "km_from_home": 10 },
		"last_transaction": { "timestamp": "2026-03-11T11:00:00Z", "km_from_current": 500 }
	}`
	var p Payload
	if err := json.Unmarshal([]byte(raw), &p); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	v := stdVectorizer().Vectorize(&p)
	// 60 minutes / 1440 = 0.041666...
	if math.Abs(v[5]-60.0/1440.0) > 1e-3 {
		t.Errorf("dim 5 minutes_since_last: got %.4f, want %.4f", v[5], 60.0/1440.0)
	}
	// 500 km / 1000 = 0.5
	if math.Abs(v[6]-0.5) > 1e-3 {
		t.Errorf("dim 6 km_from_last: got %.4f, want 0.5", v[6])
	}
	// unknown mcc 9999 -> default 0.5
	if math.Abs(v[12]-0.5) > 1e-3 {
		t.Errorf("dim 12 mcc_risk default: got %.4f, want 0.5", v[12])
	}
	// is_online true -> 1, card_present false -> 0
	if v[9] != 1 || v[10] != 0 {
		t.Errorf("dims 9,10: got %.0f,%.0f want 1,0", v[9], v[10])
	}
}
