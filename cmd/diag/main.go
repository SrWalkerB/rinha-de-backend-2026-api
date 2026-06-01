// Command diag is an OFFLINE failure-diagnosis harness (not part of the API).
//
// It replays the labeled test-data.json (the exact set the Rinha engine grades
// against) through three search methods and attributes every misclassification
// ("failure") to a cause, so we can decide a fix with data instead of guessing:
//
//   - float64-exact 5-NN  (our Vectorize + raw float64 refs)  -> vectorization fidelity
//   - uint8-brute  5-NN   (knn.Index, no Build)               -> quantization cost
//   - IVF approx          (knn.Index, Build ivf)              -> recall cost (what ships)
//
// Ground truth per entry is expected_approved (organizers labeled with exact
// brute-force float64 5-NN over references). A "failure" = approved != expected.
//
// Runs full-CPU on the dev box (no container limits apply here). It only READS;
// it changes nothing in production. See docs/performance/07-diagnostico-failures.md.
package main

import (
	"bufio"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"runtime"
	"strings"
	"sync"
	"time"

	"rinha-fraud/internal/knn"
	"rinha-fraud/internal/model"
	"rinha-fraud/internal/vectorize"
)

const dims = vectorize.Dims

func env(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

// loadModelOrNil loads the Caminho C GBDT for the "model" method; nil (skipped)
// if absent/invalid.
func loadModelOrNil(path string) *model.Model {
	b, err := os.ReadFile(path)
	if err != nil {
		log.Printf("model: %v (método model pulado)", err)
		return nil
	}
	m, err := model.LoadModel(b)
	if err != nil {
		log.Printf("model: %v (método model pulado)", err)
		return nil
	}
	return m
}

func main() {
	refPath := env("REFERENCES_PATH", "./resources/references.json.gz")
	testPath := env("TESTDATA_PATH", "../rinha-de-backend-2026/test/test-data.json")
	normPath := env("NORM_PATH", "./resources/normalization.json")
	mccPath := env("MCC_PATH", "./resources/mcc_risk.json")

	log.Printf("cores=%d", runtime.NumCPU())

	vec := loadVectorizer(normPath, mccPath)

	// Fast path: just the GBDT model vs the test-data 5-NN labels (no refs/oracle).
	if os.Getenv("MODEL_ONLY") != "" {
		mdl := loadModelOrNil(env("MODEL_PATH", "./resources/model.json"))
		if mdl == nil {
			log.Fatal("MODEL_ONLY set but no model loaded")
		}
		tf := loadTestData(testPath)
		mm := len(tf.Entries)
		log.Printf("MODEL_ONLY: %d entries — tau-sweep (baked tau=%.3f)", mm, mdl.Tau())
		mp := make([]float64, mm)
		exp := make([]bool, mm)
		for i := range tf.Entries {
			e := &tf.Entries[i]
			mp[i] = mdl.Score(vec.Vectorize(&e.Request))
			exp[i] = e.ExpectedApproved
		}
		fmt.Println("tau      FP     FN  failures   fail%%   det_score")
		for _, tau := range []float64{0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60} {
			fp, fn := 0, 0
			for i := 0; i < mm; i++ {
				ap := mp[i] < tau
				if ap == exp[i] {
					continue
				}
				if ap {
					fn++
				} else {
					fp++
				}
			}
			fail := fp + fn
			fmt.Printf("%.2f   %5d  %5d  %7d  %6.3f%%  %+8.1f\n",
				tau, fp, fn, fail, 100*float64(fail)/float64(mm), detScore(fp, fn, 0, mm))
		}
		fmt.Println("(comparar com IVF×10000 do Caminho A = 34 failures / +2482.7)")
		return
	}

	// --- load references: raw float64 (exact oracle) + two quantized indexes ---
	log.Printf("loading references from %s ...", refPath)
	t0 := time.Now()
	refsF, labels, checksum := loadRefs(refPath)
	n := len(labels)
	log.Printf("loaded %d refs in %s (sha256=%s)", n, time.Since(t0), checksum[:16]+"...")

	idxBrute := knn.NewIndex(n) // never Built -> Score does exact uint8 brute force
	idxIVF := knn.NewIndex(n)
	for r := 0; r < n; r++ {
		var v [dims]float64
		copy(v[:], refsF[r*dims:(r+1)*dims])
		idxBrute.Add(v, labels[r])
		idxIVF.Add(v, labels[r])
	}
	log.Printf("building IVF (nlist=4096 nprobe=12) ...")
	t1 := time.Now()
	idxIVF.Build(knn.BuildConfig{Mode: "ivf", NList: 4096, NProbe: 12, Iters: 8})
	log.Printf("IVF built in %s", time.Since(t1))

	// Caminho C: trained GBDT (optional). The "model" method below uses it.
	mdl := loadModelOrNil(env("MODEL_PATH", "./resources/model.json"))
	if mdl != nil {
		log.Printf("model (GBDT) loaded; baked tau=%.3f", mdl.Tau())
	}

	// --- load + vectorize the labeled test data ---
	tf := loadTestData(testPath)
	m := len(tf.Entries)
	log.Printf("test-data: %d entries (stats.total=%d edge_cases=%d)", m, tf.Stats.Total, tf.Stats.EdgeCaseCount)
	if tf.ReferencesChecksum != "" {
		if tf.ReferencesChecksum == checksum {
			log.Printf("checksum: MATCH — our references match the labeled set")
		} else {
			log.Printf("checksum: differ (ours=%s.. theirs=%s..) — may hash a different representation; not necessarily a problem",
				checksum[:12], tf.ReferencesChecksum[:12])
		}
	}

	qs := make([][dims]float64, m)
	expApproved := make([]bool, m)
	expCnt := make([]uint8, m) // expected fraud count among 5
	for i := range tf.Entries {
		e := &tf.Entries[i]
		qs[i] = vec.Vectorize(&e.Request)
		expApproved[i] = e.ExpectedApproved
		expCnt[i] = uint8(math.Round(e.ExpectedFraudScore * float64(knn.K)))
	}

	// --- per-entry fraud counts under each method (parallel over entries) ---
	f64Cnt := make([]uint8, m)
	u8Cnt := make([]uint8, m)
	ivfCnt := make([]uint8, m)
	modelP := make([]float64, m) // GBDT P(fraud); 0 if no model
	log.Printf("scoring %d entries (float64-exact + u16-brute + IVF + model) ...", m)
	t2 := time.Now()
	parallelChunks(m, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			q := qs[i]
			f64Cnt[i] = uint8(bruteF64(refsF, labels, n, &q))
			u8Cnt[i] = uint8(math.Round(idxBrute.Score(q) * float64(knn.K)))
			ivfCnt[i] = uint8(math.Round(idxIVF.Score(q) * float64(knn.K)))
			if mdl != nil {
				modelP[i] = mdl.Score(q)
			}
		}
	})
	log.Printf("scored in %s", time.Since(t2))

	approved := func(cnt uint8) bool { return float64(cnt)/float64(knn.K) < knn.Threshold }

	// --- confusion + detection_score per method vs expected ---
	fmt.Println("\n================ DIAGNÓSTICO DE FAILURES ================")
	fmt.Printf("N = %d entries\n", m)
	fmt.Println("\nMétodo            TP     TN     FP     FN   failures   fail%%   det_score")
	for _, mth := range []struct {
		name string
		cnt  []uint8
	}{
		{"float64-exact", f64Cnt},
		{"u16/f64-brute", u8Cnt},
		{"IVF (4096/12)", ivfCnt},
	} {
		tp, tn, fp, fn := 0, 0, 0, 0
		for i := 0; i < m; i++ {
			ap := approved(mth.cnt[i])
			switch {
			case ap == expApproved[i] && expApproved[i]:
				tn++
			case ap == expApproved[i]:
				tp++
			case ap: // approved a fraud
				fn++
			default: // denied a legit
				fp++
			}
		}
		fail := fp + fn
		fmt.Printf("%s  %5d  %5d  %5d  %5d  %7d  %6.3f%%  %+8.1f\n",
			mth.name, tp, tn, fp, fn, fail, 100*float64(fail)/float64(m), detScore(fp, fn, 0, m))
	}

	// --- attribution of the SCORED failures (IVF != expected) ---
	var vecFail, quantFail, recallFail int
	type flip struct {
		id            string
		exp, f64, ivf uint8
	}
	var samples []flip
	for i := 0; i < m; i++ {
		if approved(ivfCnt[i]) == expApproved[i] {
			continue // IVF got it right -> not a scored failure
		}
		switch {
		case approved(f64Cnt[i]) != expApproved[i]:
			vecFail++ // even our exact float64 disagrees -> vectorization/precision
		case approved(u8Cnt[i]) != expApproved[i]:
			quantFail++ // float64 ok but uint8 flips it -> quantization
		default:
			recallFail++ // uint8 ok but IVF flips it -> recall (the approximation)
		}
		if len(samples) < 15 {
			samples = append(samples, flip{tf.Entries[i].Request.ID, expCnt[i], f64Cnt[i], ivfCnt[i]})
		}
	}
	totFail := vecFail + quantFail + recallFail
	fmt.Println("\n---- atribuição das failures pontuadas (IVF != expected) ----")
	fmt.Printf("total            : %d\n", totFail)
	pct := func(x int) float64 {
		if totFail == 0 {
			return 0
		}
		return 100 * float64(x) / float64(totFail)
	}
	fmt.Printf("vetorização      : %4d (%.1f%%)  -> custo ZERO de latência se corrigir\n", vecFail, pct(vecFail))
	fmt.Printf("quantização u16  : %4d (%.1f%%)  -> precisão dos buckets\n", quantFail, pct(quantFail))
	fmt.Printf("recall do IVF    : %4d (%.1f%%)  -> nprobe/nlist (custa p99)\n", recallFail, pct(recallFail))

	// --- boundary concentration (failures por contagem esperada de fraude) ---
	var bucket [6]int // expected fraud count 0..5 among the IVF failures
	for i := 0; i < m; i++ {
		if approved(ivfCnt[i]) != expApproved[i] {
			bucket[expCnt[i]]++
		}
	}
	fmt.Println("\n---- failures por contagem de fraude esperada (fronteira = 2 e 3) ----")
	for c := 0; c <= 5; c++ {
		mark := ""
		if c == 2 || c == 3 {
			mark = "  <- fronteira"
		}
		fmt.Printf("  %d/5 (score %.1f): %4d%s\n", c, float64(c)/5, bucket[c], mark)
	}

	// --- vectorization fidelity: fraud_score exato float64 vs expected_fraud_score ---
	exactMatch := 0
	for i := 0; i < m; i++ {
		if f64Cnt[i] == expCnt[i] {
			exactMatch++
		}
	}
	fmt.Printf("\n---- fidelidade da vetorização ----\n")
	fmt.Printf("fraud_score float64-exato == expected_fraud_score: %d/%d (%.3f%%)\n",
		exactMatch, m, 100*float64(exactMatch)/float64(m))

	// --- exemplos de flips ----
	fmt.Println("\n---- exemplos de entries que o IVF erra (exp / f64 / ivf, contagem de fraude) ----")
	for _, s := range samples {
		fmt.Printf("  %-16s exp=%d  f64=%d  ivf=%d\n", s.id, s.exp, s.f64, s.ivf)
	}

	// --- sweep nprobe (nlist=4096) : FP/FN vs custo proxy (linhas/query) ----
	fmt.Println("\n---- sweep IVF: recall vs custo (linhas varridas/query ≈ nprobe·n/nlist) ----")
	fmt.Println("config              FP     FN  failures   fail%%   det_score   linhas/query")
	sweepIVF(idxIVF, qs, expApproved, approved, 4096, []int{8, 12, 16, 24, 32}, n)

	// nlist=8192 (rebuild) at nprobe 12 e 24
	log.Printf("building IVF nlist=8192 for the sweep ...")
	idx8192 := knn.NewIndex(n)
	for r := 0; r < n; r++ {
		var v [dims]float64
		copy(v[:], refsF[r*dims:(r+1)*dims])
		idx8192.Add(v, labels[r])
	}
	idx8192.Build(knn.BuildConfig{Mode: "ivf", NList: 8192, NProbe: 12, Iters: 8})
	sweepIVF(idx8192, qs, expApproved, approved, 8192, []int{12, 24}, n)

	// --- método MODEL (GBDT): sweep de tau (approved = P(fraude) < tau) ----
	if mdl != nil {
		fmt.Println("\n================ MÉTODO MODEL (GBDT, Caminho C) ================")
		fmt.Println("(detecção; p99 medido no docker. tau calibra a fronteira approved<tau)")
		fmt.Println("tau      FP     FN  failures   fail%%   det_score")
		bestTau, bestDet, bestFail := 0.0, math.Inf(-1), 0
		for _, tau := range []float64{0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60} {
			fp, fn := 0, 0
			for i := 0; i < m; i++ {
				ap := modelP[i] < tau
				if ap == expApproved[i] {
					continue
				}
				if ap {
					fn++
				} else {
					fp++
				}
			}
			fail := fp + fn
			det := detScore(fp, fn, 0, m)
			fmt.Printf("%.2f   %5d  %5d  %7d  %6.3f%%  %+8.1f\n",
				tau, fp, fn, fail, 100*float64(fail)/float64(m), det)
			if det > bestDet {
				bestDet, bestTau, bestFail = det, tau, fail
			}
		}
		fmt.Printf("melhor: tau=%.2f  failures=%d  det_score=%+.1f  (vs IVF×10000 = 34 / +2482.7)\n",
			bestTau, bestFail, bestDet)
		fmt.Println("nota: tau varrido sobre o test-data (calibração de 1 escalar); confirmar em held-out.")
	}

	fmt.Println("\n(p99 é medido separadamente com run-test.ps1; aqui o proxy de custo é linhas/query)")
}

// sweepIVF re-runs the IVF index at several nprobe values (cheap retune) and
// prints FP/FN + the analytic cost proxy (rows scanned per query).
func sweepIVF(ix *knn.Index, qs [][dims]float64, exp []bool, approved func(uint8) bool, nlist int, nprobes []int, n int) {
	m := len(qs)
	for _, np := range nprobes {
		ix.SetNProbe(np)
		fp, fn := 0, 0
		cnts := make([]uint8, m)
		parallelChunks(m, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				cnts[i] = uint8(math.Round(ix.Score(qs[i]) * float64(knn.K)))
			}
		})
		for i := 0; i < m; i++ {
			ap := approved(cnts[i])
			if ap == exp[i] {
				continue
			}
			if ap {
				fn++
			} else {
				fp++
			}
		}
		fail := fp + fn
		rowsPerQ := np * (n / nlist)
		fmt.Printf("nlist=%-5d nprobe=%-3d %5d  %5d  %7d  %6.3f%%  %+8.1f  %8d\n",
			nlist, np, fp, fn, fail, 100*float64(fail)/float64(m), detScore(fp, fn, 0, m), rowsPerQ)
	}
}

// bruteF64 returns the number of fraud labels among the exact 5 nearest refs to
// q, using float64 euclidean distance — the organizers' ground-truth method.
func bruteF64(refs []float64, labels []bool, n int, q *[dims]float64) int {
	var d [knn.K]float64
	var f [knn.K]bool
	for i := range d {
		d[i] = math.MaxFloat64
	}
	for r := 0; r < n; r++ {
		off := r * dims
		var dist float64
		for k := 0; k < dims; k++ {
			diff := q[k] - refs[off+k]
			dist += diff * diff
		}
		if dist >= d[knn.K-1] {
			continue
		}
		pos := knn.K - 1
		for pos > 0 && d[pos-1] > dist {
			d[pos] = d[pos-1]
			f[pos] = f[pos-1]
			pos--
		}
		d[pos] = dist
		f[pos] = labels[r]
	}
	cnt := 0
	for _, fr := range f {
		if fr {
			cnt++
		}
	}
	return cnt
}

// detScore implements the AVALIACAO.md detection_score for a given confusion.
func detScore(fp, fn, errs, N int) float64 {
	fail := float64(fp+fn+errs) / float64(N)
	if fail > 0.15 {
		return -3000
	}
	E := float64(fp + 3*fn + 5*errs)
	eps := E / float64(N)
	if eps < 0.001 {
		eps = 0.001
	}
	return 1000*math.Log10(1/eps) - 300*math.Log10(1+E)
}

func parallelChunks(n int, fn func(lo, hi int)) {
	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = 1
	}
	if workers > n {
		workers = n
	}
	if workers <= 1 {
		fn(0, n)
		return
	}
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		lo := w * chunk
		if lo >= n {
			break
		}
		hi := lo + chunk
		if hi > n {
			hi = n
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			fn(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}

// ---- loaders ----

func loadVectorizer(normPath, mccPath string) *vectorize.Vectorizer {
	var nf struct {
		MaxAmount            float64 `json:"max_amount"`
		MaxInstallments      float64 `json:"max_installments"`
		AmountVsAvgRatio     float64 `json:"amount_vs_avg_ratio"`
		MaxMinutes           float64 `json:"max_minutes"`
		MaxKm                float64 `json:"max_km"`
		MaxTxCount24h        float64 `json:"max_tx_count_24h"`
		MaxMerchantAvgAmount float64 `json:"max_merchant_avg_amount"`
	}
	mustJSON(normPath, &nf)
	mcc := map[string]float64{}
	mustJSON(mccPath, &mcc)
	return vectorize.New(vectorize.Norm{
		MaxAmount:            nf.MaxAmount,
		MaxInstallments:      nf.MaxInstallments,
		AmountVsAvgRatio:     nf.AmountVsAvgRatio,
		MaxMinutes:           nf.MaxMinutes,
		MaxKm:                nf.MaxKm,
		MaxTxCount24h:        nf.MaxTxCount24h,
		MaxMerchantAvgAmount: nf.MaxMerchantAvgAmount,
	}, mcc)
}

func mustJSON(path string, v any) {
	b, err := os.ReadFile(path)
	if err != nil {
		log.Fatalf("read %s: %v", path, err)
	}
	if err := json.Unmarshal(b, v); err != nil {
		log.Fatalf("parse %s: %v", path, err)
	}
}

type refRecord struct {
	Vector [dims]float64 `json:"vector"`
	Label  string        `json:"label"`
}

// loadRefs streams references.json(.gz) into a flat float64 slice + fraud labels,
// hashing the decompressed bytes (to compare with test-data's checksum).
func loadRefs(path string) (refs []float64, labels []bool, checksum string) {
	f, err := os.Open(path)
	if err != nil {
		log.Fatalf("open %s: %v", path, err)
	}
	defer f.Close()

	var r io.Reader = bufio.NewReaderSize(f, 1<<20)
	if strings.HasSuffix(path, ".gz") {
		gz, err := gzip.NewReader(r)
		if err != nil {
			log.Fatalf("gzip %s: %v", path, err)
		}
		defer gz.Close()
		r = gz
	}
	h := sha256.New()
	dec := json.NewDecoder(io.TeeReader(r, h))

	refs = make([]float64, 0, 3_000_000*dims)
	labels = make([]bool, 0, 3_000_000)
	if _, err := dec.Token(); err != nil { // opening '['
		log.Fatalf("read array start: %v", err)
	}
	for dec.More() {
		var rec refRecord
		if err := dec.Decode(&rec); err != nil {
			log.Fatalf("decode record %d: %v", len(labels), err)
		}
		refs = append(refs, rec.Vector[:]...)
		labels = append(labels, rec.Label == "fraud")
	}
	// Hash covers the bytes the decoder consumed (essentially the whole file);
	// the checksum compare is best-effort (the preimage convention is unknown).
	return refs, labels, hex.EncodeToString(h.Sum(nil))
}

type testData struct {
	ReferencesChecksum string `json:"references_checksum_sha256"`
	Stats              struct {
		Total         int `json:"total"`
		EdgeCaseCount int `json:"edge_case_count"`
	} `json:"stats"`
	Entries []struct {
		Request            vectorize.Payload `json:"request"`
		ExpectedApproved   bool              `json:"expected_approved"`
		ExpectedFraudScore float64           `json:"expected_fraud_score"`
	} `json:"entries"`
}

func loadTestData(path string) *testData {
	b, err := os.ReadFile(path)
	if err != nil {
		log.Fatalf("read %s: %v", path, err)
	}
	var td testData
	if err := json.Unmarshal(b, &td); err != nil {
		log.Fatalf("parse %s: %v", path, err)
	}
	return &td
}
