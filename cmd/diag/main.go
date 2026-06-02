// Command diag is the offline correctness + latency harness for the partitioned
// index. It replays the official test set (test-data.json) through:
//
//  1. the partitioned Score (production path), and
//  2. BruteScore (the exact float64 oracle — the ground truth),
//
// asserting they make identical decisions (proving detection is preserved), then
// reports the detection result vs expected_approved and the per-query work
// (rows scanned + wall-clock), which is the offline latency proxy.
//
// Usage (Go not on PATH on Windows: prepend C:\Program Files\Go\bin):
//
//	go run ./cmd/diag
//
// Env: REFERENCES_PATH, TESTDATA_PATH, NORM_PATH, MCC_PATH, BRUTE_SAMPLE.
package main

import (
	"encoding/json"
	"log"
	"math"
	"os"
	"runtime"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"rinha-fraud/internal/dataset"
	"rinha-fraud/internal/index"
	"rinha-fraud/internal/vectorize"
)

type entry struct {
	Request          vectorize.Payload `json:"request"`
	ExpectedApproved bool              `json:"expected_approved"`
}

type testData struct {
	Entries []entry `json:"entries"`
}

func env(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func envi(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func envf(key string, def float64) float64 {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return def
}

func main() {
	refPath := env("REFERENCES_PATH", "./resources/references.json.gz")
	tdPath := env("TESTDATA_PATH", "../rinha-de-backend-2026/test/test-data.json")
	normPath := env("NORM_PATH", "./resources/normalization.json")
	mccPath := env("MCC_PATH", "./resources/mcc_risk.json")
	bruteSample := envi("BRUTE_SAMPLE", 3000)

	// Vectorizer (same constants as the server).
	nb, err := os.ReadFile(normPath)
	if err != nil {
		log.Fatalf("read %s: %v", normPath, err)
	}
	mb, err := os.ReadFile(mccPath)
	if err != nil {
		log.Fatalf("read %s: %v", mccPath, err)
	}
	vec, err := vectorize.Parse(nb, mb)
	if err != nil {
		log.Fatalf("parse vectorizer: %v", err)
	}

	// Partitioned index: load a prebuilt index.bin if INDEX_PATH is set (fast,
	// lets slack sweeps skip the rebuild), else build from references.
	t0 := time.Now()
	var ix *index.Index
	if p := os.Getenv("INDEX_PATH"); p != "" {
		ix, err = index.LoadIndex(p)
		if err != nil {
			log.Fatalf("load index %s: %v", p, err)
		}
		log.Printf("loaded prebuilt index %s (%d vectors) in %s", p, ix.Len(), time.Since(t0))
	} else {
		nlist := envi("INDEX_NLIST", index.DefaultNList)
		iters := envi("INDEX_KMEANS_ITERS", index.DefaultKMeansIters)
		log.Printf("loading + IVF-partitioning references from %s (nlist=%d) ...", refPath, nlist)
		ix, err = dataset.Load(refPath, 3_000_000, nlist, iters)
		if err != nil {
			log.Fatalf("load references: %v", err)
		}
		log.Printf("index over %d vectors in %s", ix.Len(), time.Since(t0))
	}
	nprobe := envi("INDEX_NPROBE", 16)
	nprobeHigh := envi("INDEX_NPROBE_HIGH", 0) // 0 = single-tier (off); sweep turns it on
	triggerRadius := envf("INDEX_TRIGGER_RADIUS", 0.98)
	triggerMargin := envf("INDEX_TRIGGER_MARGIN", -1)
	ix.SetNProbe(nprobe)
	ix.SetMaxScan(envi("INDEX_MAX_SCAN", 0))
	ix.SetTriggerRadius(triggerRadius)
	ix.SetTriggerMargin(triggerMargin)
	ix.SetNProbeHigh(nprobeHigh) // after SetNProbe: HIGH compared to cheap nprobe
	ix.SetNProbeHighForCount(2, envi("INDEX_NPROBE_HIGH_C2", nprobeHigh))
	ix.SetNProbeHighForCount(3, envi("INDEX_NPROBE_HIGH_C3", nprobeHigh))
	ix.SetNProbeHighForCount(4, envi("INDEX_NPROBE_HIGH_C4", nprobeHigh))
	log.Printf("nlist=%d nprobe=%d nprobeHigh=%d triggerRadius=%.3f triggerMargin=%.2f maxScan=%d",
		ix.NList(), nprobe, nprobeHigh, triggerRadius, triggerMargin, envi("INDEX_MAX_SCAN", 0))

	// Test set.
	tb, err := os.ReadFile(tdPath)
	if err != nil {
		log.Fatalf("read %s: %v", tdPath, err)
	}
	var td testData
	if err := json.Unmarshal(tb, &td); err != nil {
		log.Fatalf("parse test data: %v", err)
	}
	n := len(td.Entries)

	// Pre-vectorize all queries.
	queries := make([][index.Dims]float64, n)
	for i := range td.Entries {
		queries[i] = vec.Vectorize(&td.Entries[i].Request)
	}

	pass1 := n
	if lim := envi("PASS1_LIMIT", 0); lim > 0 && lim < n {
		pass1 = lim
	}
	log.Printf("replaying %d test entries (pass1=%d)", n, pass1)

	// --- Pass 1: partitioned Score over entries (failures, latency, scan) ---
	var fp, fn int
	var escalated int
	var vote [index.K + 1]voteStats
	scanned := make([]int, pass1)
	centEv := make([]int, pass1)
	latNs := make([]int64, pass1)
	start := time.Now()
	for i := 0; i < pass1; i++ {
		q0 := time.Now()
		score, sc, tr := ix.ScoreScanTrace(queries[i])
		latNs[i] = time.Since(q0).Nanoseconds()
		scanned[i] = sc
		centEv[i] = tr.CheapCentroids + tr.HighCentroids
		vs := &vote[tr.CheapFraudCount]
		vs.total++
		vs.scanned = append(vs.scanned, sc)
		if tr.Escalated {
			escalated++
			vs.escalated++
			vs.highRows += tr.HighScanned
		}
		approved := score < index.Threshold
		if approved != td.Entries[i].ExpectedApproved {
			if approved { // approved a fraud → false negative
				fn++
				vs.fn++
			} else { // denied a legit → false positive
				fp++
				vs.fp++
			}
		}
	}
	totalElapsed := time.Since(start)

	// --- Pass 2: brute oracle on a sample: assert identical decisions + measure
	// the K-th-NN squared distance (the search radius the index must cover). ---
	step := 1
	if bruteSample > 0 && bruteSample < n {
		step = n / bruteSample
	}
	var mismatch int64
	var wg sync.WaitGroup
	sem := make(chan struct{}, runtime.NumCPU())
	var mu sync.Mutex
	var kth []float64
	for i := 0; i < n; i += step {
		sem <- struct{}{}
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			defer func() { <-sem }()
			b, kd := ix.BruteDebug(queries[i])
			a, _ := ix.ScoreScan(queries[i])
			if (a < index.Threshold) != (b < index.Threshold) || a != b {
				atomic.AddInt64(&mismatch, 1)
			}
			mu.Lock()
			kth = append(kth, kd)
			mu.Unlock()
		}(i)
	}
	wg.Wait()

	// --- Report ---
	E := fp + 3*fn
	failures := fp + fn
	eps := float64(E) / float64(pass1)
	failRate := float64(failures) / float64(pass1)
	detScore := detectionScore(E, eps)
	meanUs := float64(totalElapsed.Microseconds()) / float64(pass1)

	sort.Ints(scanned)
	sort.Slice(latNs, func(i, j int) bool { return latNs[i] < latNs[j] })
	sort.Float64s(kth)

	log.Printf("================ DETECTION (partitioned vs expected, n=%d) ==========", pass1)
	log.Printf("  FP=%d  FN=%d  failures=%d  E=%d  eps=%.6f  failure_rate=%.6f%%",
		fp, fn, failures, E, eps, failRate*100)
	log.Printf("  detection_score ~= %.1f  (cap +3000 at E=0)", detScore)
	log.Printf("  escalated=%d  (%.2f%% of queries hit the high-nprobe pass)",
		escalated, float64(escalated)/float64(pass1)*100)
	log.Printf("================ CHEAP VOTE BREAKDOWN ==============================")
	for i := 0; i <= index.K; i++ {
		vs := &vote[i]
		if vs.total == 0 {
			continue
		}
		sort.Ints(vs.scanned)
		log.Printf("  cheapCount=%d total=%d escalated=%d fp=%d fn=%d highRowsMean=%.0f rowsP99=%d",
			i, vs.total, vs.escalated, vs.fp, vs.fn, safeMeanRows(vs.highRows, vs.escalated), vs.scanned[pct(len(vs.scanned), 99)])
	}
	log.Printf("================ EXACTNESS (partitioned vs brute oracle) ============")
	log.Printf("  checked=%d  mismatches=%d  (want 0 => partitioned == exact 5-NN)", len(kth), mismatch)
	log.Printf("================ 5th-NN squared distance (search radius) ============")
	log.Printf("  p50=%.4f  p90=%.4f  p99=%.4f  max=%.4f  (>1.0 => crosses a bucket)",
		kth[pct(len(kth), 50)], kth[pct(len(kth), 90)], kth[pct(len(kth), 99)], kth[len(kth)-1])
	log.Printf("================ LATENCY (per query, dev CPU, single-threaded) ======")
	log.Printf("  mean=%.2fus  p50=%.2fus  p99=%.2fus  max=%.2fus",
		meanUs, us(latNs[pass1/2]), us(latNs[pct(pass1, 99)]), us(latNs[pass1-1]))
	log.Printf("================ WORK (reference rows scanned per query) ============")
	log.Printf("  mean=%.0f  p50=%d  p99=%d  max=%d",
		mean(scanned), scanned[pass1/2], scanned[pct(pass1, 99)], scanned[pass1-1])
	sort.Ints(centEv)
	log.Printf("================ CENTROID EVALS per query (nlist scans, NOT in WORK) =")
	log.Printf("  mean=%.0f  p50=%d  p99=%d  max=%d",
		mean(centEv), centEv[pass1/2], centEv[pct(pass1, 99)], centEv[pass1-1])
}

type voteStats struct {
	total     int
	escalated int
	fp        int
	fn        int
	highRows  int
	scanned   []int
}

func us(ns int64) float64 { return float64(ns) / 1000 }

func pct(n, p int) int {
	i := n * p / 100
	if i >= n {
		i = n - 1
	}
	return i
}

func mean(xs []int) float64 {
	var s float64
	for _, x := range xs {
		s += float64(x)
	}
	return s / float64(len(xs))
}

func safeMeanRows(rows, n int) float64 {
	if n == 0 {
		return 0
	}
	return float64(rows) / float64(n)
}

// detectionScore mirrors AVALIACAO.md: 1000·log10(1/max(eps,0.001)) − 300·log10(1+E).
func detectionScore(E int, eps float64) float64 {
	e := eps
	if e < 0.001 {
		e = 0.001
	}
	return 1000*math.Log10(1/e) - 300*math.Log10(1+float64(E))
}
