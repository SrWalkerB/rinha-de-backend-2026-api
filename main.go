// Command rinha-fraud is the fraud-detection API for the Rinha de Backend 2026.
// It exposes GET /ready and POST /fraud-score on the configured address.
package main

import (
	"embed"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"strconv"
	"sync/atomic"
	"time"

	// Registers /debug/pprof/* handlers on http.DefaultServeMux. They are only
	// served when PPROF_ADDR is set (see main); production/submission leaves it
	// unset, so this is inert there.
	_ "net/http/pprof"

	"rinha-fraud/internal/dataset"
	"rinha-fraud/internal/knn"
	"rinha-fraud/internal/vectorize"
)

//go:embed resources/normalization.json resources/mcc_risk.json
var resourcesFS embed.FS

type server struct {
	vec   *vectorize.Vectorizer
	index atomic.Pointer[knn.Index]
	ready atomic.Bool
}

type response struct {
	Approved   bool    `json:"approved"`
	FraudScore float64 `json:"fraud_score"`
}

// fallback is returned on any error: a fast 200 avoids the heavily-weighted
// HTTP-error penalty (AVALIACAO.md) at the cost of a possible FP/FN.
var fallback = response{Approved: true, FraudScore: 0.0}

func (s *server) handleReady(w http.ResponseWriter, _ *http.Request) {
	if s.ready.Load() {
		w.WriteHeader(http.StatusOK)
		return
	}
	w.WriteHeader(http.StatusServiceUnavailable)
}

func (s *server) handleScore(w http.ResponseWriter, r *http.Request) {
	defer func() {
		if rec := recover(); rec != nil {
			log.Printf("recovered in handleScore: %v", rec)
			writeJSON(w, fallback)
		}
	}()

	var p vectorize.Payload
	if err := json.NewDecoder(r.Body).Decode(&p); err != nil {
		writeJSON(w, fallback)
		return
	}

	ix := s.index.Load()
	if ix == nil {
		writeJSON(w, fallback)
		return
	}

	score := ix.Score(s.vec.Vectorize(&p))
	writeJSON(w, response{Approved: score < knn.Threshold, FraudScore: score})
}

func writeJSON(w http.ResponseWriter, resp response) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(resp)
}

// loadVectorizer builds the Vectorizer from the embedded constant files.
func loadVectorizer() (*vectorize.Vectorizer, error) {
	var nf struct {
		MaxAmount            float64 `json:"max_amount"`
		MaxInstallments      float64 `json:"max_installments"`
		AmountVsAvgRatio     float64 `json:"amount_vs_avg_ratio"`
		MaxMinutes           float64 `json:"max_minutes"`
		MaxKm                float64 `json:"max_km"`
		MaxTxCount24h        float64 `json:"max_tx_count_24h"`
		MaxMerchantAvgAmount float64 `json:"max_merchant_avg_amount"`
	}
	nb, err := resourcesFS.ReadFile("resources/normalization.json")
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(nb, &nf); err != nil {
		return nil, err
	}

	mb, err := resourcesFS.ReadFile("resources/mcc_risk.json")
	if err != nil {
		return nil, err
	}
	mcc := map[string]float64{}
	if err := json.Unmarshal(mb, &mcc); err != nil {
		return nil, err
	}

	norm := vectorize.Norm{
		MaxAmount:            nf.MaxAmount,
		MaxInstallments:      nf.MaxInstallments,
		AmountVsAvgRatio:     nf.AmountVsAvgRatio,
		MaxMinutes:           nf.MaxMinutes,
		MaxKm:                nf.MaxKm,
		MaxTxCount24h:        nf.MaxTxCount24h,
		MaxMerchantAvgAmount: nf.MaxMerchantAvgAmount,
	}
	return vectorize.New(norm, mcc), nil
}

func getenv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func atoiEnv(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func main() {
	vec, err := loadVectorizer()
	if err != nil {
		log.Fatalf("load constants: %v", err)
	}
	s := &server{vec: vec}

	refPath := getenv("REFERENCES_PATH", "./resources/references.json.gz")
	capHint := atoiEnv("REFERENCES_CAPACITY", 3_000_000)
	go func() {
		// Fast path: a pre-built index baked at image-build time (step 05). Load
		// it and skip k-means entirely. KNN_NPROBE retunes it without rebuilding.
		if path := getenv("INDEX_PATH", ""); path != "" {
			t0 := time.Now()
			ix, err := knn.LoadIndex(path)
			if err == nil {
				ix.SetNProbe(atoiEnv("KNN_NPROBE", 0)) // 0 => keep the baked value
				s.index.Store(ix)
				s.ready.Store(true)
				log.Printf("ready: loaded prebuilt index %s (%d vectors) in %s",
					path, ix.Len(), time.Since(t0))
				return
			}
			log.Printf("prebuilt index %s unavailable (%v); building from references", path, err)
		}

		// Fallback: load the dataset and build the index at startup.
		log.Printf("loading references from %s ...", refPath)
		ix, err := dataset.Load(refPath, capHint)
		if err != nil {
			log.Fatalf("load references: %v", err)
		}

		cfg := knn.BuildConfig{
			Mode:   getenv("KNN_INDEX", "ivf"),
			NList:  atoiEnv("KNN_NLIST", 256),
			NProbe: atoiEnv("KNN_NPROBE", 8),
			Iters:  atoiEnv("KNN_KMEANS_ITERS", 8),
		}
		log.Printf("building %q index over %d vectors (nlist=%d nprobe=%d) ...",
			cfg.Mode, ix.Len(), cfg.NList, cfg.NProbe)
		t0 := time.Now()
		ix.Build(cfg)
		log.Printf("index built in %s", time.Since(t0))

		s.index.Store(ix)
		s.ready.Store(true)
		log.Printf("ready: %d reference vectors loaded", ix.Len())
	}()

	// Optional pprof debug server on a SEPARATE port, off unless PPROF_ADDR is
	// set. Passing nil serves http.DefaultServeMux, where net/http/pprof
	// registered its handlers. Never expose this on the API port.
	if pprofAddr := getenv("PPROF_ADDR", ""); pprofAddr != "" {
		go func() {
			log.Printf("pprof listening on %s (e.g. http://%s/debug/pprof/)", pprofAddr, pprofAddr)
			if err := http.ListenAndServe(pprofAddr, nil); err != nil {
				log.Printf("pprof server: %v", err)
			}
		}()
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /ready", s.handleReady)
	mux.HandleFunc("POST /fraud-score", s.handleScore)

	addr := getenv("ADDR", ":9999")
	log.Printf("listening on %s", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatalf("server: %v", err)
	}
}
