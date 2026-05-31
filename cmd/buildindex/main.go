// Command buildindex constructs the IVF search index offline — at image-build
// time, with the full machine (no CPU cap) — and writes it to a binary file.
// At startup the API loads that file instead of running k-means under the
// runtime cap, which makes /ready fast and lets nlist be large (smaller cells
// => faster query => lower p99). See
// docs/performance/05-preprocessamento-no-build.md.
package main

import (
	"log"
	"os"
	"strconv"
	"time"

	"rinha-fraud/internal/dataset"
	"rinha-fraud/internal/knn"
)

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

func main() {
	ref := env("REFERENCES_PATH", "./resources/references.json.gz")
	out := env("INDEX_OUT", "./resources/index.bin")
	capHint := envi("REFERENCES_CAPACITY", 3_000_000)
	cfg := knn.BuildConfig{
		Mode:   "ivf",
		NList:  envi("KNN_NLIST", 4096),
		NProbe: envi("KNN_NPROBE", 8),
		Iters:  envi("KNN_KMEANS_ITERS", 8),
	}

	log.Printf("buildindex: loading references from %s ...", ref)
	t0 := time.Now()
	ix, err := dataset.Load(ref, capHint)
	if err != nil {
		log.Fatalf("load references: %v", err)
	}
	log.Printf("loaded %d vectors in %s", ix.Len(), time.Since(t0))

	log.Printf("building ivf index (nlist=%d nprobe=%d iters=%d) ...",
		cfg.NList, cfg.NProbe, cfg.Iters)
	t1 := time.Now()
	ix.Build(cfg)
	log.Printf("index built in %s", time.Since(t1))

	if err := ix.Save(out); err != nil {
		log.Fatalf("save index: %v", err)
	}
	if fi, err := os.Stat(out); err == nil {
		log.Printf("wrote %s (%.1f MB)", out, float64(fi.Size())/(1<<20))
	} else {
		log.Printf("wrote %s", out)
	}
}
