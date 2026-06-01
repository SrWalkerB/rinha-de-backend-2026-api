// Command genlabels produces 5-NN IMITATION labels for training the Caminho C
// GBDT — so the model learns the actual 5-NN decision (the gabarito) instead of
// the noisy individual reference labels (which floored detection at ~1.85%, see
// docs/performance/JORNADA-E-DECISOES.md §9).
//
// Method (leave-one-out, no self-match): the first half of the references is the
// KNN DATABASE; each point in the second half is labeled by its 5-NN fraud_score
// over that database (fast IVF). Query points are NOT in the database, so there
// is no distance-0 self-match biasing the label toward the point's own class.
//
// Output: a compact little-endian binary, one example per 15 float32:
//
//	[ v0 .. v13 (the 14-dim vector), fivenn_fraud_score ]
//
// train/train.py reads it (LABELS_BIN env) and trains on y = (score >= 0.6),
// i.e. the 5-NN "denied" decision. Offline only; nothing here ships in the image.
package main

import (
	"bufio"
	"compress/gzip"
	"encoding/binary"
	"encoding/json"
	"io"
	"log"
	"math"
	"os"
	"runtime"
	"strings"
	"sync"
	"time"

	"rinha-fraud/internal/knn"
	"rinha-fraud/internal/vectorize"
)

const dims = vectorize.Dims

func env(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func main() {
	refPath := env("REFERENCES_PATH", "./resources/references.json.gz")
	outPath := env("LABELS_OUT", "./resources/train_labels.bin")
	log.Printf("cores=%d", runtime.NumCPU())

	log.Printf("loading references from %s ...", refPath)
	t0 := time.Now()
	refs, labels := loadRefs(refPath)
	n := len(labels)
	log.Printf("loaded %d refs in %s", n, time.Since(t0))

	half := n / 2
	log.Printf("building IVF database over first %d refs ...", half)
	t1 := time.Now()
	db := knn.NewIndex(half)
	for r := 0; r < half; r++ {
		var v [dims]float64
		copy(v[:], refs[r*dims:(r+1)*dims])
		db.Add(v, labels[r])
	}
	db.Build(knn.BuildConfig{Mode: "ivf", NList: 2048, NProbe: 16, Iters: 8})
	log.Printf("database built in %s", time.Since(t1))

	qN := n - half
	out := make([]float32, qN*(dims+1))
	log.Printf("labeling %d query refs by 5-NN over the database ...", qN)
	t2 := time.Now()
	parallelChunks(qN, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			row := half + i
			var v [dims]float64
			copy(v[:], refs[row*dims:(row+1)*dims])
			score := db.Score(v) // 5-NN fraud fraction over the database (no self)
			base := i * (dims + 1)
			for d := 0; d < dims; d++ {
				out[base+d] = float32(v[d])
			}
			out[base+dims] = float32(score)
		}
	})
	log.Printf("labeled in %s", time.Since(t2))

	if err := writeF32(outPath, out); err != nil {
		log.Fatalf("write %s: %v", outPath, err)
	}
	fi, _ := os.Stat(outPath)
	log.Printf("wrote %s (%d examples, %.1f MB)", outPath, qN, float64(fi.Size())/(1<<20))
}

func writeF32(path string, s []float32) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriterSize(f, 1<<20)
	var b [4]byte
	for _, v := range s {
		binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
		if _, err := w.Write(b[:]); err != nil {
			return err
		}
	}
	return w.Flush()
}

type refRecord struct {
	Vector [dims]float64 `json:"vector"`
	Label  string        `json:"label"`
}

func loadRefs(path string) (refs []float64, labels []bool) {
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
	dec := json.NewDecoder(r)
	refs = make([]float64, 0, 3_000_000*dims)
	labels = make([]bool, 0, 3_000_000)
	if _, err := dec.Token(); err != nil {
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
	return refs, labels
}

func parallelChunks(n int, fn func(lo, hi int)) {
	workers := runtime.GOMAXPROCS(0)
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
