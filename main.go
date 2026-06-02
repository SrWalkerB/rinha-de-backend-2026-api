// Command rinha-fraud is the fraud-detection API for the Rinha de Backend 2026.
// It exposes GET /ready and POST /fraud-score on the configured address.
package main

import (
	"bytes"
	"embed"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"strconv"
	"sync"
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

// There are only K+1 possible answers: fraud_score = n/K for a fraud count n in
// 0..K, with approved = (score < Threshold). Pre-render each one ONCE so the hot
// path writes bytes with zero allocation. They are built with json.Marshal (plus
// the trailing '\n' json.Encoder.Encode used to add), so the wire bytes stay
// byte-identical to the old encoder. Package init runs before tests, too.
var responseBody = buildResponses()

// fallbackBody is returned on any error: a fast 200 avoids the heavily-weighted
// HTTP-error penalty (AVALIACAO.md) at the cost of a possible FP/FN. It is the
// n=0 answer: {approved:true, fraud_score:0}.
var fallbackBody = responseBody[0]

func buildResponses() [index.K + 1][]byte {
	var bodies [index.K + 1][]byte
	for n := 0; n <= index.K; n++ {
		score := float64(n) / float64(index.K)
		b, err := json.Marshal(response{Approved: score < index.Threshold, FraudScore: score})
		if err != nil {
			panic(err) // a bool+float struct cannot fail to marshal
		}
		bodies[n] = append(b, '\n')
	}
	return bodies
}

// contentTypeJSON is assigned straight into the response header map (instead of
// Header().Set) to avoid the one-element slice Set allocates per call.
var contentTypeJSON = []string{"application/json"}

// bufPool reuses request-body buffers so the heap stays flat: with GOGC=off the
// GOMEMLIMIT pacer then never arms, removing the GC/CFS-throttle p99 stalls.
var bufPool = sync.Pool{New: func() any { return new(bytes.Buffer) }}

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
			writeBody(w, fallbackBody)
		}
	}()

	buf := bufPool.Get().(*bytes.Buffer)
	buf.Reset()
	defer bufPool.Put(buf)
	if _, err := buf.ReadFrom(r.Body); err != nil {
		writeBody(w, fallbackBody)
		return
	}

	q, err := s.vec.VectorizeJSON(buf.Bytes())
	if err != nil {
		writeBody(w, fallbackBody)
		return
	}

	ix := s.ix.Load()
	if ix == nil {
		writeBody(w, fallbackBody)
		return
	}

	writeBody(w, responseBody[ix.ScoreCount(q)])
}

func writeBody(w http.ResponseWriter, body []byte) {
	w.Header()["Content-Type"] = contentTypeJSON
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(body)
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

func atofEnv(key string, def float64) float64 {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
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
	nprobe := atoiEnv("INDEX_NPROBE", 16)   // cheap-tier cells scanned per bucket per query
	maxScan := atoiEnv("INDEX_MAX_SCAN", 0) // 0 = unlimited; tail guardrail
	// Adaptive (two-tier) nprobe: a query whose cheap-pass 5th-NN radius reaches
	// TRIGGER_RADIUS re-runs at NPROBE_HIGH (drives the cross-bucket misses to 0
	// while only escalating the rare borderline query). HIGH≤NPROBE disables it.
	nprobeHigh := atoiEnv("INDEX_NPROBE_HIGH", 192)
	// Margin gate (primary): escalate when the cheap vote is within this many votes
	// of the 0.6 boundary (count 2..4). Measured E=0 at ~3% escalation. Radius is the
	// fallback gate, used only when margin<0.
	triggerMargin := atofEnv("INDEX_TRIGGER_MARGIN", 1)
	triggerRadius := atofEnv("INDEX_TRIGGER_RADIUS", 0.98)
	nprobeHighC2 := atoiEnv("INDEX_NPROBE_HIGH_C2", nprobeHigh)
	nprobeHighC3 := atoiEnv("INDEX_NPROBE_HIGH_C3", nprobeHigh)
	nprobeHighC4 := atoiEnv("INDEX_NPROBE_HIGH_C4", nprobeHigh)

	tune := func(ix *index.Index) {
		ix.SetNProbe(nprobe)
		ix.SetMaxScan(maxScan)
		ix.SetTriggerRadius(triggerRadius)
		ix.SetTriggerMargin(triggerMargin)
		ix.SetNProbeHigh(nprobeHigh) // after SetNProbe: HIGH is compared to the cheap nprobe
		ix.SetNProbeHighForCount(2, nprobeHighC2)
		ix.SetNProbeHighForCount(3, nprobeHighC3)
		ix.SetNProbeHighForCount(4, nprobeHighC4)
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
				log.Printf("ready: loaded prebuilt index %s (%d vectors, nlist=%d nprobe=%d nprobeHigh=%d triggerRadius=%.3f) in %s",
					path, ix.Len(), ix.NList(), nprobe, nprobeHigh, triggerRadius, time.Since(t0))
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
	// Explicit timeouts bound the tail and protect against slow/stuck peers. The
	// IdleTimeout sits ABOVE nginx's upstream keepalive idle so nginx recycles
	// pooled connections, never the api mid-flight (a premature api-side close
	// forces nginx to reconnect, adding latency).
	srv := &http.Server{
		Addr:              addr,
		Handler:           mux,
		ReadTimeout:       2 * time.Second,
		ReadHeaderTimeout: 1 * time.Second,
		WriteTimeout:      2 * time.Second,
		IdleTimeout:       65 * time.Second,
		MaxHeaderBytes:    1 << 14,
	}
	log.Printf("listening on %s", addr)
	if err := srv.ListenAndServe(); err != nil {
		log.Fatalf("server: %v", err)
	}
}
