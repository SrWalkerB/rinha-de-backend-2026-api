package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"rinha-fraud/internal/knn"
	"rinha-fraud/internal/vectorize"
)

func testVectorizer() *vectorize.Vectorizer {
	norm := vectorize.Norm{
		MaxAmount: 10000, MaxInstallments: 12, AmountVsAvgRatio: 10,
		MaxMinutes: 1440, MaxKm: 1000, MaxTxCount24h: 20, MaxMerchantAvgAmount: 10000,
	}
	mcc := map[string]float64{"7802": 0.75, "5411": 0.15}
	return vectorize.New(norm, mcc)
}

const fraudBody = `{
	"id": "tx-3330991687",
	"transaction": { "amount": 9505.97, "installments": 10, "requested_at": "2026-03-14T05:15:12Z" },
	"customer": { "avg_amount": 81.28, "tx_count_24h": 20, "known_merchants": ["MERC-008"] },
	"merchant": { "id": "MERC-068", "mcc": "7802", "avg_amount": 54.86 },
	"terminal": { "is_online": false, "card_present": true, "km_from_home": 952.27 },
	"last_transaction": null
}`

// serverWithFraudNeighbors builds a server whose 5 nearest neighbors for the
// fraud payload are all fraud (Score == 1.0).
func serverWithFraudNeighbors(t *testing.T) *server {
	t.Helper()
	vec := testVectorizer()
	var p vectorize.Payload
	if err := json.Unmarshal([]byte(fraudBody), &p); err != nil {
		t.Fatal(err)
	}
	fv := vec.Vectorize(&p)
	ix := knn.NewIndex(8)
	for i := 0; i < 5; i++ {
		ix.Add(fv, true)
	}
	s := &server{vec: vec}
	s.index.Store(ix)
	s.ready.Store(true)
	return s
}

func TestReadyWhenLoaded(t *testing.T) {
	s := serverWithFraudNeighbors(t)
	rr := httptest.NewRecorder()
	s.handleReady(rr, httptest.NewRequest(http.MethodGet, "/ready", nil))
	if rr.Code != http.StatusOK {
		t.Errorf("ready loaded: code = %d, want 200", rr.Code)
	}
}

func TestReadyWhenNotLoaded(t *testing.T) {
	s := &server{vec: testVectorizer()} // ready defaults to false, no index
	rr := httptest.NewRecorder()
	s.handleReady(rr, httptest.NewRequest(http.MethodGet, "/ready", nil))
	if rr.Code < 500 {
		t.Errorf("ready not loaded: code = %d, want 5xx", rr.Code)
	}
}

func TestScoreFraudPayload(t *testing.T) {
	s := serverWithFraudNeighbors(t)
	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/fraud-score", strings.NewReader(fraudBody))
	s.handleScore(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("code = %d, want 200", rr.Code)
	}
	var resp struct {
		Approved   bool    `json:"approved"`
		FraudScore float64 `json:"fraud_score"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("bad json: %v (%s)", err, rr.Body.String())
	}
	if resp.FraudScore != 1.0 {
		t.Errorf("fraud_score = %v, want 1.0", resp.FraudScore)
	}
	if resp.Approved {
		t.Errorf("approved = true, want false for fraud_score 1.0")
	}
}

// Malformed body must still return 200 with a safe fallback (avoid HTTP error
// penalty, which AVALIACAO.md weights heaviest).
func TestScoreMalformedBodyFallsBack(t *testing.T) {
	s := serverWithFraudNeighbors(t)
	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/fraud-score", strings.NewReader("{not json"))
	s.handleScore(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("code = %d, want 200 (fallback)", rr.Code)
	}
	var resp struct {
		Approved   bool    `json:"approved"`
		FraudScore float64 `json:"fraud_score"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("bad json: %v", err)
	}
	if !resp.Approved || resp.FraudScore != 0.0 {
		t.Errorf("fallback = %+v, want {approved:true, fraud_score:0}", resp)
	}
}
