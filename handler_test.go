package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"rinha-fraud/internal/index"
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
func serverWithFraudNeighbors(tb testing.TB) *server {
	tb.Helper()
	vec := testVectorizer()
	var p vectorize.Payload
	if err := json.Unmarshal([]byte(fraudBody), &p); err != nil {
		tb.Fatal(err)
	}
	fv := vec.Vectorize(&p)
	b := index.NewBuilder(8, 1, 1)
	for i := 0; i < 5; i++ {
		b.Add(fv, true)
	}
	s := &server{vec: vec}
	s.ix.Store(b.Build())
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

// TestPrecomputedResponsesMatchJSON locks the pre-rendered bodies to exactly what
// the old json.Encoder.Encode path produced (json.Marshal + trailing '\n'), so the
// wire format can never silently drift from the precompute.
func TestPrecomputedResponsesMatchJSON(t *testing.T) {
	for n := 0; n <= index.K; n++ {
		score := float64(n) / float64(index.K)
		want, err := json.Marshal(response{Approved: score < index.Threshold, FraudScore: score})
		if err != nil {
			t.Fatal(err)
		}
		want = append(want, '\n')
		if !bytes.Equal(responseBody[n], want) {
			t.Errorf("responseBody[%d] = %q, want %q", n, responseBody[n], want)
		}
	}
	if !bytes.Equal(fallbackBody, responseBody[0]) {
		t.Errorf("fallbackBody = %q, want responseBody[0] = %q", fallbackBody, responseBody[0])
	}
}

// nopRW is a zero-allocation ResponseWriter stub for alloc assertions.
type nopRW struct{ hdr http.Header }

func (w *nopRW) Header() http.Header         { return w.hdr }
func (w *nopRW) Write(b []byte) (int, error) { return len(b), nil }
func (w *nopRW) WriteHeader(int)             {}

// TestWriteBodyZeroAlloc proves the response path allocates nothing: the bytes are
// precomputed and Content-Type is assigned to an already-present header key.
func TestWriteBodyZeroAlloc(t *testing.T) {
	w := &nopRW{hdr: http.Header{}}
	allocs := testing.AllocsPerRun(1000, func() {
		writeBody(w, responseBody[index.K])
	})
	if allocs != 0 {
		t.Errorf("writeBody allocs/op = %v, want 0", allocs)
	}
}

// TestHandlerPoolReuseNoBleed drives several payloads through the same (pooled)
// server back-to-back, interleaving a malformed body, and checks each answer
// independently — catching any stale-field bleed from Payload/buffer reuse.
func TestHandlerPoolReuseNoBleed(t *testing.T) {
	s := serverWithFraudNeighbors(t)
	for i := 0; i < 3; i++ {
		rr := httptest.NewRecorder()
		s.handleScore(rr, httptest.NewRequest(http.MethodPost, "/fraud-score", strings.NewReader(fraudBody)))
		var resp response
		if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
			t.Fatalf("iter %d: bad json: %v", i, err)
		}
		if resp.FraudScore != 1.0 || resp.Approved {
			t.Errorf("iter %d: got %+v, want {false, 1.0}", i, resp)
		}

		// A malformed body in between must fall back AND must not corrupt the pool.
		rr2 := httptest.NewRecorder()
		s.handleScore(rr2, httptest.NewRequest(http.MethodPost, "/fraud-score", strings.NewReader("{bad")))
		var fb response
		if err := json.Unmarshal(rr2.Body.Bytes(), &fb); err != nil {
			t.Fatalf("iter %d: bad fallback json: %v", i, err)
		}
		if !fb.Approved || fb.FraudScore != 0.0 {
			t.Errorf("iter %d: fallback = %+v, want {true, 0}", i, fb)
		}
	}
}

// BenchmarkHandleScore measures the full request path's allocs/op (run with
// -benchmem). It rewinds one *bytes.Reader per iteration, so the reported allocs
// reflect the handler itself — the residual json.Unmarshal string allocs are the
// signal for whether Fase 2 (hand parser) is needed.
func BenchmarkHandleScore(b *testing.B) {
	s := serverWithFraudNeighbors(b)
	body := []byte(fraudBody)
	r := bytes.NewReader(body)
	req := httptest.NewRequest(http.MethodPost, "/fraud-score", r)
	w := &nopRW{hdr: http.Header{}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r.Reset(body)
		s.handleScore(w, req)
	}
}
