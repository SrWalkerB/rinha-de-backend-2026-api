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
	"rinha-fraud/internal/index"
	"rinha-fraud/internal/vectorize"
)

//go:embed resources/normalization.json resources/mcc_risk.json
var resourcesFS embed.FS

type server struct {
	vec   *vectorize.Vectorizer
	ix    atomic.Pointer[index.Index]
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

	ix := s.ix.Load()
	if ix == nil {
		writeJSON(w, fallback)
		return
	}

	score := ix.Score(s.vec.Vectorize(&p))
	writeJSON(w, response{Approved: score < index.Threshold, FraudScore: score})
}

func writeJSON(w http.ResponseWriter, resp response) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(resp)
}

// loadVectorizer builds the Vectorizer from the embedded constant files.
func loadVectorizer() (*vectorize.Vectorizer, error) {
	nb, err := resourcesFS.ReadFile("resources/normalization.json")
	if err != nil {
		return nil, err
	}
	mb, err := resourcesFS.ReadFile("resources/mcc_risk.json")
	if err != nil {
		return nil, err
	}
	return vectorize.Parse(nb, mb)
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
	nprobe := atoiEnv("INDEX_NPROBE", 8)    // cells scanned per bucket per query
	maxScan := atoiEnv("INDEX_MAX_SCAN", 0) // 0 = unlimited; tail guardrail

	tune := func(ix *index.Index) {
		ix.SetNProbe(nprobe)
		ix.SetMaxScan(maxScan)
	}

	go func() {
		// Fast path: an IVF index baked at image-build time. Load it and be ready
		// in tens of ms.
		if path := getenv("INDEX_PATH", ""); path != "" {
			t0 := time.Now()
			ix, err := index.LoadIndex(path)
			if err == nil {
				tune(ix)
				s.ix.Store(ix)
				s.ready.Store(true)
				log.Printf("ready: loaded prebuilt index %s (%d vectors, nlist=%d nprobe=%d) in %s",
					path, ix.Len(), ix.NList(), nprobe, time.Since(t0))
				return
			}
			log.Printf("prebuilt index %s unavailable (%v); building from references", path, err)
		}

		// Fallback: build the index from the dataset at startup.
		nlist := atoiEnv("INDEX_NLIST", index.DefaultNList)
		iters := atoiEnv("INDEX_KMEANS_ITERS", index.DefaultKMeansIters)
		log.Printf("loading references from %s (nlist=%d) ...", refPath, nlist)
		t0 := time.Now()
		ix, err := dataset.Load(refPath, capHint, nlist, iters)
		if err != nil {
			log.Fatalf("load references: %v", err)
		}
		tune(ix)
		s.ix.Store(ix)
		s.ready.Store(true)
		log.Printf("ready: built index over %d reference vectors in %s", ix.Len(), time.Since(t0))
	}()

	// Optional pprof debug server on a SEPARATE port, off unless PPROF_ADDR is
	// set. Never expose this on the API port.
	if pprofAddr := getenv("PPROF_ADDR", ""); pprofAddr != "" {
		go func() {
			log.Printf("pprof listening on %s", pprofAddr)
			if err := http.ListenAndServe(pprofAddr, nil); err != nil {
				log.Printf("pprof server: %v", err)
			}
		}()
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /ready", s.handleReady)
	mux.HandleFunc("POST /fraud-score", s.handleScore)

	addr := getenv("ADDR", ":8080")
	log.Printf("listening on %s", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatalf("server: %v", err)
	}
}
